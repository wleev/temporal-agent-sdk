package model_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/wleev/temporal-agent-sdk/model"
)

func TestMediaBlockConstructors(t *testing.T) {
	img := model.ImageBlock("image/png", []byte{1, 2, 3})
	require.Equal(t, model.BlockMedia, img.Kind)
	require.Equal(t, "image/png", img.MIMEType)
	require.Equal(t, []byte{1, 2, 3}, img.Data)

	aud := model.AudioBlock("audio/wav", []byte{4, 5})
	require.Equal(t, model.BlockMedia, aud.Kind)
	require.Equal(t, "audio/wav", aud.MIMEType)
}

func TestUserContent(t *testing.T) {
	m := model.UserContent(model.TextBlock("what is this?"), model.ImageBlock("image/png", []byte{9}))
	require.Equal(t, model.RoleUser, m.Role)
	require.Len(t, m.Blocks, 2)
	require.Equal(t, model.BlockText, m.Blocks[0].Kind)
	require.Equal(t, model.BlockMedia, m.Blocks[1].Kind)
}

// Text() concatenates text blocks only, so media in a mixed message never leaks
// into a text rendering.
func TestText_IgnoresMedia(t *testing.T) {
	m := model.UserContent(
		model.TextBlock("look:"),
		model.ImageBlock("image/png", []byte{1, 2, 3}),
		model.TextBlock(" and this"),
	)
	require.Equal(t, "look: and this", m.Text())
}

// A media block must survive workflow history: the bytes are base64 in JSON and
// come back byte-for-byte.
func TestMediaBlock_JSONRoundTrip(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47, 0x00, 0xff}
	m := model.UserContent(model.ImageBlock("image/png", data))

	raw, err := json.Marshal(m)
	require.NoError(t, err)
	require.Contains(t, string(raw), `"mime_type":"image/png"`)

	var back model.Message
	require.NoError(t, json.Unmarshal(raw, &back))
	require.Equal(t, data, back.Blocks[0].Data)
	require.Equal(t, "image/png", back.Blocks[0].MIMEType)
}

func TestMediaPlaceholder(t *testing.T) {
	require.Contains(t, model.MediaPlaceholder(model.ImageBlock("image/png", nil)), "image omitted")
	require.Contains(t, model.MediaPlaceholder(model.AudioBlock("audio/wav", nil)), "audio omitted")
	require.Contains(t, model.MediaPlaceholder(model.Block{Kind: model.BlockMedia, MIMEType: "video/mp4"}), "video omitted")
	require.Contains(t, model.MediaPlaceholder(model.Block{Kind: model.BlockMedia, MIMEType: "application/pdf"}), "application/pdf")
}
