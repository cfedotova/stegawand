let coverDataURL = null;

const drop = $("drop");
const fileInput = $("file");
const runBtn = $("run");
const newImgBtn = $("newImg");
const statusEl = $("status");

bindDropzone(drop, fileInput, (url) => {
  coverDataURL = url;
  $("i_cover").src = url;
  setStatus(statusEl, "image loaded — press “run full demo”");
});

async function loadSample(random=false) {
  setStatus(statusEl, "loading sample image…");
  try {
    const r = await fetch("/api/sample-image" + (random ? "?random=1" : ""));
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    coverDataURL = j.image;
    $("i_cover").src = j.image;
    setStatus(statusEl, `loaded sample: ${j.name} — press “run full demo”`);
  } catch (err) {
    setStatus(statusEl, "couldn’t load sample: " + err.message + " — drop your own", false);
  }
}

newImgBtn.addEventListener("click", () => loadSample(true));

runBtn.addEventListener("click", async () => {
  if (!coverDataURL) { setStatus(statusEl, "no image yet", false); return; }
  runBtn.disabled = true; newImgBtn.disabled = true;
  setStatus(statusEl, "running full pipeline (encode → 4 distortions → decode)…");
  try {
    const payload = {
      image: coverDataURL,
      message: $("message").value || "hello",
      distortions: {
        blur: true, noise: true, grayscale: true, jpeg: true,
        noise_sigma: 0.02, blur_sigma: 1.0, blur_kernel: 5, jpeg_quality: 75,
      },
    };
    const data = await postJson("/api/encode", payload);
    renderAll(data);
    setStatus(statusEl, "done — scroll through sections 1–4 to screenshot");
  } catch (err) {
    setStatus(statusEl, "error: " + err.message, false);
  } finally {
    runBtn.disabled = false; newImgBtn.disabled = false;
  }
});

function renderAll(data) {
  $("i_cover").src = data.cover;
  $("i_stego").src = data.stego;
  $("i_res").src   = data.residual;
  $("psnr_line").textContent =
    `PSNR cover/stego: ${data.psnr} dB · message used: "${data.message_used}"`;
  const dl = $("dl_stego");
  dl.href = data.stego;
  dl.hidden = false;

  const clean = data.results.find(r => r.name === "clean") || data.results[0];
  renderBitStream($("b_encoded"), data.encoded_bits, data.encoded_bits);
  renderBitStream($("b_decoded"), clean.decoded_bits, data.encoded_bits);

  const grid = $("dgrid");
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

  const params = {
    clean:     "—",
    noise:     "σ = 0.02",
    blur:      "5 × 5, σ = 1.0",
    grayscale: "BT.601 luma",
    jpeg:      "q = 75 (fake JPEG)",
  };
  const tbody = $("summary_table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of data.results) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(r.name)}</td>
      <td>${escapeHtml(params[r.name] || "—")}</td>
      <td><span class="bit-acc-pill ${colorAcc(r.bit_acc)}">${(r.bit_acc*100).toFixed(1)}%</span></td>
      <td><code>${escapeHtml(r.decoded || "—")}</code></td>
    `;
    tbody.appendChild(tr);
  }
  $("footnote").hidden = false;
}

loadSample(false);
