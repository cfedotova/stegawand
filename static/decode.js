let dataURL = null;

const drop = $("drop");
const fileInput = $("file");
const runBtn = $("run");
const statusEl = $("status");
const bitWrap = $("bit_compare_wrap");

bindDropzone(drop, fileInput, (url) => {
  dataURL = url;
  $("i_input").src = url;
  setStatus(statusEl, "image loaded");
});

runBtn.addEventListener("click", async () => {
  if (!dataURL) { setStatus(statusEl, "drop an image first", false); return; }
  const payload = {
    image: dataURL,
    distortions: {
      blur: $("d_blur").checked,
      noise: $("d_noise").checked,
      grayscale: $("d_gray").checked,
      jpeg: $("d_jpeg").checked,
    },
  };
  runBtn.disabled = true;
  setStatus(statusEl, "decoding…");
  try {
    const data = await postJson("/api/decode", payload);
    $("i_input").src = data.input;
    $("decoded_text").textContent = data.decoded || "—";
    bitWrap.hidden = false;
    renderBitStream($("decoded_bits"), data.decoded_bits, null);
    setStatus(statusEl, "done");
  } catch (err) {
    setStatus(statusEl, "error: " + err.message, false);
  } finally {
    runBtn.disabled = false;
  }
});
