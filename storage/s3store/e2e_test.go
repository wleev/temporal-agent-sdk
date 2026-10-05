package s3store_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
	enumspb "go.temporal.io/api/enums/v1"
	historypb "go.temporal.io/api/history/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"
	"google.golang.org/protobuf/proto"

	"github.com/wleev/temporal-agent-sdk/agent"
	"github.com/wleev/temporal-agent-sdk/agenttest"
	"github.com/wleev/temporal-agent-sdk/model"
	"github.com/wleev/temporal-agent-sdk/plugin"
	"github.com/wleev/temporal-agent-sdk/storage"
	"github.com/wleev/temporal-agent-sdk/storage/external"
	"github.com/wleev/temporal-agent-sdk/storage/s3store"
)

const (
	// seaweedImage pins the SeaweedFS release the container runs.
	seaweedImage = "chrislusf/seaweedfs:4.47"
	// seaweedS3Port is the container port serving the S3 API.
	seaweedS3Port = "8333/tcp"
	// seaweedAccessKey and seaweedSecretKey are the S3 credentials the container
	// is started with.
	seaweedAccessKey = "agentsdk"
	seaweedSecretKey = "agentsdk-secret"

	seaweedWorkflow  = "s3store-e2e"
	seaweedThreshold = 4 << 10
)

// startSeaweedFS runs a SeaweedFS testcontainer for the duration of t and
// returns an S3 client for it. It skips t when no container runtime is
// available.
func startSeaweedFS(t *testing.T) *s3.Client {
	t.Helper()
	testcontainers.SkipIfProviderIsNotHealthy(t)

	container, err := testcontainers.Run(t.Context(), seaweedImage,
		testcontainers.WithCmd(
			"mini",
			"-dir=/data",
			"-admin.ui=false",
			"-master.telemetry=false",
			"-master.volumeSizeLimitMB=32",
			"-volume.max=16",
		),
		testcontainers.WithTmpfs(map[string]string{"/data": "rw,size=512m"}),
		testcontainers.WithEnv(map[string]string{
			"AWS_ACCESS_KEY_ID":     seaweedAccessKey,
			"AWS_SECRET_ACCESS_KEY": seaweedSecretKey,
		}),
		testcontainers.WithExposedPorts(seaweedS3Port),
		testcontainers.WithWaitStrategy(
			wait.ForHTTP("/healthz").
				WithPort(seaweedS3Port).
				WithStartupTimeout(time.Minute),
		),
	)
	testcontainers.CleanupContainer(t, container)
	require.NoError(t, err, "start SeaweedFS container")

	endpoint, err := container.PortEndpoint(t.Context(), seaweedS3Port, "http")
	require.NoError(t, err, "SeaweedFS S3 endpoint")

	cfg := aws.Config{
		Region: "us-east-1",
		Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: seaweedAccessKey, SecretAccessKey: seaweedSecretKey}, nil
		}),
	}
	return s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.BaseEndpoint = aws.String(endpoint)
		o.UsePathStyle = true
	})
}

// createBucket creates a uniquely named bucket and returns its name.
func createBucket(t *testing.T, c *s3.Client) string {
	t.Helper()
	bucket := fmt.Sprintf("agentsdk-%d", time.Now().UnixNano())
	_, err := c.CreateBucket(t.Context(), &s3.CreateBucketInput{Bucket: aws.String(bucket)})
	require.NoError(t, err, "create bucket %q", bucket)
	return bucket
}

func seaweedAgentWorkflow(ctx workflow.Context) (*agent.Result, error) {
	a, err := agent.NewAgent("assistant", "test-model")
	if err != nil {
		return nil, err
	}
	return agent.Run(ctx, a, "Write a long report.")
}

// TestSeaweedFS runs the store against a SeaweedFS container.
func TestSeaweedFS(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SeaweedFS e2e: needs a container runtime (-short)")
	}
	s3Client := startSeaweedFS(t)

	t.Run("round trip and missing objects", func(t *testing.T) {
		store, err := s3store.New(s3Client, createBucket(t, s3Client))
		require.NoError(t, err)

		data := []byte(strings.Repeat("payload bytes ", 10000))
		require.NoError(t, store.Put(t.Context(), "prefix/object", data))
		require.NoError(t, store.Put(t.Context(), "prefix/object", data), "a repeated Put must succeed")

		got, err := store.Get(t.Context(), "prefix/object")
		require.NoError(t, err)
		assert.Equal(t, data, got)

		_, err = store.Get(t.Context(), "prefix/absent")
		assert.ErrorIs(t, err, storage.ErrNotFound)
	})

	t.Run("missing bucket is not a missing object", func(t *testing.T) {
		store, err := s3store.New(s3Client, "agentsdk-no-such-bucket")
		require.NoError(t, err)

		_, err = store.Get(t.Context(), "prefix/object")
		require.Error(t, err)
		assert.NotErrorIs(t, err, storage.ErrNotFound)
	})

	// An agent whose model reply exceeds the threshold returns the full reply,
	// the reply is an object in the bucket, no history event reaches the
	// threshold, and a replayer reading the bucket replays the history.
	t.Run("agent run", func(t *testing.T) {
		bucket := createBucket(t, s3Client)
		store, err := s3store.New(s3Client, bucket)
		require.NoError(t, err)
		ext, err := external.New(store, storage.WithThreshold(seaweedThreshold))
		require.NoError(t, err)
		storagePlugin, err := external.NewPlugin(ext)
		require.NoError(t, err)

		report := strings.Repeat("A quarterly figure worth reporting. ", 4000)
		fake := agenttest.NewFakeProvider(agenttest.Says(report))
		sdkPlugin, err := plugin.New(plugin.Config{Providers: []model.Provider{fake}})
		require.NoError(t, err)

		ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
		defer cancel()
		srv, err := testsuite.StartDevServer(ctx, testsuite.DevServerOptions{LogLevel: "error"})
		if err != nil {
			t.Skipf("skipping: could not start dev server: %v", err)
		}
		t.Cleanup(func() { _ = srv.Stop() })

		c, err := client.Dial(client.Options{
			HostPort: srv.FrontendHostPort(),
			Plugins:  []client.Plugin{storagePlugin},
		})
		require.NoError(t, err)
		t.Cleanup(c.Close)

		const tq = "agentsdk-s3store-test"
		w := worker.New(c, tq, worker.Options{Plugins: []worker.Plugin{sdkPlugin}})
		w.RegisterWorkflowWithOptions(seaweedAgentWorkflow, workflow.RegisterOptions{Name: seaweedWorkflow})
		require.NoError(t, w.Start())
		t.Cleanup(w.Stop)

		run, err := c.ExecuteWorkflow(ctx, client.StartWorkflowOptions{TaskQueue: tq}, seaweedWorkflow)
		require.NoError(t, err)
		var res agent.Result
		require.NoError(t, run.Get(ctx, &res))
		assert.Equal(t, report, res.Output)

		listed, err := s3Client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
			Bucket: aws.String(bucket),
			Prefix: aws.String(storage.DefaultKeyPrefix),
		})
		require.NoError(t, err)
		assert.NotEmpty(t, listed.Contents, "the oversized reply must be an object under %s", storage.DefaultKeyPrefix)

		// A client without external storage reads the history as recorded.
		history := &historypb.History{}
		iter := srv.Client().GetWorkflowHistory(ctx, run.GetID(), run.GetRunID(), false,
			enumspb.HISTORY_EVENT_FILTER_TYPE_ALL_EVENT)
		for iter.HasNext() {
			ev, err := iter.Next()
			require.NoError(t, err)
			history.Events = append(history.Events, ev)
		}
		for _, ev := range history.Events {
			size := proto.Size(ev)
			assert.Less(t, size, seaweedThreshold, "event %d (%s) is %d bytes; oversized payloads must be references",
				ev.GetEventId(), ev.GetEventType(), size)
		}

		replayer, err := worker.NewWorkflowReplayerWithOptions(worker.WorkflowReplayerOptions{
			Plugins: []worker.Plugin{storagePlugin},
		})
		require.NoError(t, err)
		replayer.RegisterWorkflowWithOptions(seaweedAgentWorkflow, workflow.RegisterOptions{Name: seaweedWorkflow})
		assert.NoError(t, replayer.ReplayWorkflowHistory(nil, history))
	})
}
