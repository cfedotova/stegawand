let coverDataURL = null;

const drop = $("drop");
const fileInput = $("file");
const runBtn = $("run");
const sweepBtn = $("sweep");
const statusEl = $("status");
const intensitySlider = $("sIntensity");
const intensityVal = $("vIntensity");
const resWrap = $("stress_result");
const chartWrap = $("sweep_chart_wrap");
const canvas = $("sweep_canvas");
const table = $("sweep_table");

bindDropzone(drop, fileInput, (url) => {
  coverDataURL = url;
  $("i_cover").src = url;
  setStatus(statusEl, "image loaded");
});

intensitySlider.addEventListener("input", () => {
  intensityVal.textContent = Number(intensitySlider.value).toFixed(2) + "×";
});

function buildPayload(intensity) {
  return {
    image: coverDataURL,
    message: $("message").value || "hello",
    intensity,
    distortions: {
      noise:     $("d_noise").checked,
      blur:      $("d_blur").checked,
      grayscale: $("d_gray").checked,
      jpeg:      $("d_jpeg").checked,
      noise_sigma:  0.02,
      blur_sigma:   1.0,
      jpeg_quality: 75,
    },
  };
}

function renderResult(data) {
  $("i_cover").src = data.cover;
  $("i_stego").src = data.stego;
  $("i_dist").src  = data.distorted;
  const dl = $("dl_stego");
  dl.href = data.stego;
  dl.hidden = false;
  resWrap.hidden = false;
  $("decoded_text").textContent = data.decoded || "—";
  $("expected_text").textContent = data.message_used || "—";
  $("bit_acc").textContent = (data.bit_acc * 100).toFixed(1) + "%";
  $("bit_acc").className = "big-acc " + colorAcc(data.bit_acc);
  $("eff_noise").textContent = `σ_noise=${data.effective.noise_sigma}`;
  $("eff_blur").textContent  = `σ_blur=${data.effective.blur_sigma}`;
  $("eff_jpeg").textContent  = `jpeg q=${data.effective.jpeg_quality}`;
  renderBitStream($("encoded_bits"), data.encoded_bits, data.encoded_bits);
  renderBitStream($("decoded_bits"), data.decoded_bits, data.encoded_bits);
}

runBtn.addEventListener("click", async () => {
  if (!coverDataURL) { setStatus(statusEl, "drop an image first", false); return; }
  runBtn.disabled = true;
  setStatus(statusEl, "running stress test…");
  try {
    const data = await postJson("/api/stress", buildPayload(parseFloat(intensitySlider.value)));
    renderResult(data);
    setStatus(statusEl, "done");
  } catch (err) {
    setStatus(statusEl, "error: " + err.message, false);
  } finally {
    runBtn.disabled = false;
  }
});

sweepBtn.addEventListener("click", async () => {
  if (!coverDataURL) { setStatus(statusEl, "drop an image first", false); return; }
  sweepBtn.disabled = true; runBtn.disabled = true;
  const intensities = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0];
  const points = [];
  chartWrap.hidden = false;
  table.innerHTML = "<thead><tr><th>intensity</th><th>bit-acc, %</th><th>σ_noise</th><th>σ_blur</th><th>jpeg q</th><th>decoded</th></tr></thead><tbody></tbody>";
  const tbody = table.querySelector("tbody");

  for (const i of intensities) {
    setStatus(statusEl, `sweeping: intensity = ${i.toFixed(2)}×…`);
    try {
      const data = await postJson("/api/stress", buildPayload(i));
      points.push({intensity: i, acc: data.bit_acc, eff: data.effective, decoded: data.decoded});
      renderResult(data);
      drawChart(points);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${i.toFixed(2)}×</td>
        <td><span class="bit-acc-pill ${colorAcc(data.bit_acc)}">${(data.bit_acc*100).toFixed(1)}%</span></td>
        <td>${data.effective.noise_sigma}</td>
        <td>${data.effective.blur_sigma}</td>
        <td>${data.effective.jpeg_quality}</td>
        <td><code>${escapeHtml(data.decoded || "—")}</code></td>
      `;
      tbody.appendChild(tr);
    } catch (err) {
      setStatus(statusEl, "sweep error: " + err.message, false);
      break;
    }
  }
  const breaks = points.find(p => p.acc < 0.9);
  setStatus(statusEl,
    breaks
      ? `threshold ≈ intensity = ${breaks.intensity.toFixed(2)}× (bit-acc dropped below 90% here)`
      : "every point > 90% — model held up across all levels");
  sweepBtn.disabled = false; runBtn.disabled = false;
});

function drawChart(points) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const padL = 50, padR = 16, padT = 18, padB = 36;
  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = "#f0d6e3"; ctx.lineWidth = 1;
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillStyle = "#8e556d";
  for (let p = 0; p <= 100; p += 20) {
    const y = padT + (1 - p / 100) * (H - padT - padB);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(p + "%", 8, y + 4);
  }
  const y90 = padT + (1 - 0.9) * (H - padT - padB);
  ctx.strokeStyle = "#d04464"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(padL, y90); ctx.lineTo(W - padR, y90); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#d04464"; ctx.fillText("90%", W - padR - 30, y90 - 5);

  ctx.fillStyle = "#8e556d";
  const xMax = 3.0;
  for (let x = 0; x <= xMax; x += 0.5) {
    const px = padL + (x / xMax) * (W - padL - padR);
    ctx.fillText(x.toFixed(1) + "×", px - 10, H - 12);
  }
  ctx.fillText("intensity", (W - padL) / 2, H - 2);

  if (points.length > 1) {
    ctx.strokeStyle = "#ff7ea7"; ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((p, i) => {
      const px = padL + (p.intensity / xMax) * (W - padL - padR);
      const py = padT + (1 - p.acc) * (H - padT - padB);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  for (const p of points) {
    const px = padL + (p.intensity / xMax) * (W - padL - padR);
    const py = padT + (1 - p.acc) * (H - padT - padB);
    ctx.fillStyle = p.acc >= 0.9 ? "#4caf7d" : (p.acc >= 0.7 ? "#d29333" : "#d04464");
    ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
  }
}
