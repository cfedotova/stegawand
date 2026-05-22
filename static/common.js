const $ = (id) => document.getElementById(id);

function setStatus(el, text, ok=true) {
  el.textContent = text || "";
  el.style.color = ok ? "" : "#d04464";
}

function colorAcc(x) {
  if (x >= 0.95) return "good";
  if (x >= 0.80) return "ok";
  return "bad";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function bindDropzone(drop, fileInput, onLoad) {
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault(); fileInput.click();
    }
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault(); drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) readFile(f);
  });
  fileInput.addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) readFile(f);
  });
  function readFile(f) {
    const r = new FileReader();
    r.onload = (ev) => onLoad(ev.target.result, f);
    r.readAsDataURL(f);
  }
}

function renderBitStream(target, decoded, encoded=null) {
  target.innerHTML = "";
  for (let i = 0; i < decoded.length; i++) {
    const b = decoded[i];
    const span = document.createElement("span");
    span.className = "bit " + (encoded
        ? (b === encoded[i] ? "match" : "flip")
        : "unknown");
    span.textContent = b;
    target.appendChild(span);
    if ((i + 1) % 8 === 0 && i < decoded.length - 1) {
      const sep = document.createElement("span");
      sep.className = "byte-sep";
      target.appendChild(sep);
    }
  }
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  let data;
  try { data = await res.json(); }
  catch { throw new Error(`server returned ${res.status} ${res.statusText}`); }
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}
