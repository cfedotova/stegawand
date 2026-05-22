let coverDataURL = null;

const drop = $("drop");
const fileInput = $("file");
const runBtn = $("run");
const statusEl = $("status");
const grid = $("distortion_grid");
const psnrLine = $("psnr_line");
const bitWrap = $("bit_compare_wrap");
const encodedBitsEl = $("encoded_bits");
const decodedStreamsEl = $("decoded_streams");

bindDropzone(drop, fileInput, (dataUrl) => {
  coverDataURL = dataUrl;
  $("i_cover").src = dataUrl;
  setStatus(statusEl, "image loaded");
});

runBtn.addEventListener("click", async () => {
  if (!coverDataURL) { setStatus(statusEl, "drop an image first", false); return; }
  const message = $("message").value || "";
  if (!message.trim()) { setStatus(statusEl, "type a secret message", false); return; }

  const payload = {
    image: coverDataURL,
    message,
    distortions: {
      blur: $("d_blur").checked,
      noise: $("d_noise").checked,
      grayscale: $("d_gray").checked,
      jpeg: $("d_jpeg").checked,
    },
  };
  runBtn.disabled = true;
  setStatus(statusEl, "encoding & decoding…");
  try {
    const data = await postJson("/api/encode", payload);

    $("i_cover").src = data.cover;
    $("i_stego").src = data.stego;
    $("i_res").src   = data.residual;
    const dl = $("download_stego");
    dl.href = data.stego;
    dl.hidden = false;
    psnrLine.textContent =
      `cover/stego PSNR: ${data.psnr} dB · message: "${data.message_used}"`;

    bitWrap.hidden = false;
    renderBitStream(encodedBitsEl, data.encoded_bits, data.encoded_bits);
    decodedStreamsEl.innerHTML = "";
    for (const r of data.results) {
      const wrap = document.createElement("div");
      wrap.className = "bit-row";
      wrap.innerHTML = `
        <span class="bit-row-label">${escapeHtml(r.name)}
          <span class="bit-acc-pill ${colorAcc(r.bit_acc)}">${(r.bit_acc*100).toFixed(1)}%</span>
        </span>
        <code class="bit-stream" data-bits="${r.decoded_bits}"></code>
      `;
      decodedStreamsEl.appendChild(wrap);
      renderBitStream(wrap.querySelector(".bit-stream"), r.decoded_bits, data.encoded_bits);
    }

    grid.innerHTML = "";
    for (const r of data.results) {
      const card = document.createElement("div");
      card.className = "card distortion";
      card.innerHTML = `
        <h3>${escapeHtml(r.name)}</h3>
        <img src="${r.image}" alt="${escapeHtml(r.name)}" />
        <p class="decoded-text">${escapeHtml(r.decoded || "—")}</p>
        <small class="bit-acc ${colorAcc(r.bit_acc)}">bit-acc ${(r.bit_acc*100).toFixed(1)}%</small>
      `;
      grid.appendChild(card);
    }
    setStatus(statusEl, "done");
  } catch (err) {
    setStatus(statusEl, "error: " + err.message, false);
  } finally {
    runBtn.disabled = false;
  }
});
