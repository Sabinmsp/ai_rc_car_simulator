// AI RC Car Sim - frontend
// ------------------------
// Talks to the backend over a single WebSocket. The backend pushes:
//   - {type:"world",  ...snapshot}    -> redraw the room canvas + debug panel
//   - {type:"chat",   role, text}      -> chat bubble in the phone panel
//   - {type:"log",    channel, text, data} -> log row in the comm log
// We send:
//   - {type:"user_message", text}
//   - {type:"stop"}
//   - {type:"reset"}

(() => {
  const $ = (id) => document.getElementById(id);

  const elStatusDot  = $("status-dot");
  const elStatusText = $("status-text");
  const btnConnect   = $("btn-connect");
  const btnReset     = $("btn-reset");
  const elChat       = $("chat");
  const elChatForm   = $("chat-form");
  const elChatInput  = $("chat-input");

  const canvas = $("room");
  const ctx = canvas.getContext("2d");

  let ws = null;
  let world = null; // last world snapshot

  // ---- WebSocket lifecycle ------------------------------------------------

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const url = `ws://${location.host}/ws`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      elStatusDot.className = "dot online";
      elStatusText.textContent = "connected";
      btnConnect.textContent = "Disconnect";
      btnReset.disabled = false;
      addSystemMsg("WebSocket connected.");
    };

    ws.onclose = () => {
      elStatusDot.className = "dot offline";
      elStatusText.textContent = "disconnected";
      btnConnect.textContent = "Connect";
      btnReset.disabled = true;
      addSystemMsg("WebSocket disconnected.");
    };

    ws.onerror = () => addSystemMsg("WebSocket error (is the backend running?).");

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      handleServerMessage(msg);
    };
  }

  function disconnect() {
    if (ws) ws.close();
  }

  function send(obj) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      addSystemMsg("Not connected. Click Connect first.");
      return;
    }
    ws.send(JSON.stringify(obj));
  }

  // ---- Inbound messages ---------------------------------------------------

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "world":
        world = msg;
        drawWorld();
        break;
      case "chat":
        addChatMsg(msg.role, msg.text);
        break;
      case "log":
        // log messages are still sent by the server for debugging but the
        // phone UI doesn't render them any more.
        break;
    }
  }

  // ---- Chat & logs --------------------------------------------------------

  function addChatMsg(role, text) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    elChat.appendChild(div);
    elChat.scrollTop = elChat.scrollHeight;
  }

  function addSystemMsg(text) {
    const div = document.createElement("div");
    div.className = "msg system";
    div.textContent = text;
    elChat.appendChild(div);
    elChat.scrollTop = elChat.scrollHeight;
  }

  // ---- Room canvas --------------------------------------------------------

  function drawWorld() {
    if (!world) return;
    const W = canvas.width, H = canvas.height;
    const N = world.grid_size;
    const cell = Math.floor(Math.min(W, H) / N);

    ctx.clearRect(0, 0, W, H);

    // background gradient
    const bg = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W * 0.75);
    bg.addColorStop(0, "#0e1320");
    bg.addColorStop(1, "#05070c");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // FOV cells: fade by distance from car
    const carCx = world.car.x * cell + cell / 2;
    const carCy = world.car.y * cell + cell / 2;
    for (const [x, y] of world.fov_cells) {
      const cx = x * cell + cell / 2;
      const cy = y * cell + cell / 2;
      const dist = Math.hypot(cx - carCx, cy - carCy);
      const maxDist = cell * 8;
      const alpha = Math.max(0.08, 0.35 * (1 - dist / maxDist));
      ctx.fillStyle = `rgba(250, 204, 21, ${alpha})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }

    // walls — brick-like look with subtle top highlight
    const walls = world.walls || [];
    for (const [x, y] of walls) {
      const px = x * cell, py = y * cell;
      const grd = ctx.createLinearGradient(px, py, px, py + cell);
      grd.addColorStop(0, "#5b6576");
      grd.addColorStop(1, "#394252");
      ctx.fillStyle = grd;
      ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
      ctx.strokeStyle = "#2a313e";
      ctx.lineWidth = 1;
      ctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3);
    }

    // grid lines (very subtle)
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= N; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, N * cell); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(N * cell, i * cell); ctx.stroke();
    }

    // objects: emojis, one per cell (car is drawn afterwards so it sits on top)
    const emojis = {
      football: "⚽",
      key:      "🔑",
      chair:    "🪑",
      dog:      "🐶",
    };
    const emojiFont = `${Math.floor(cell * 0.75)}px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif`;
    ctx.font = emojiFont;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const o of world.objects) {
      const cx = o.x * cell + cell / 2;
      const cy = o.y * cell + cell / 2;
      ctx.fillText(emojis[o.name] || "❓", cx, cy);
    }

    // car (emoji)
    drawCar(ctx, world.car, cell);
  }

  function drawCar(ctx, car, cell) {
    const cx = car.x * cell + cell / 2;
    const cy = car.y * cell + cell / 2;

    ctx.save();
    ctx.translate(cx, cy);
    // Car glyph is always drawn facing up, regardless of car.dir.
    // (The FOV cone on the canvas still reflects the true heading.)
    ctx.rotate(-Math.PI / 2);
    ctx.font = `${Math.floor(cell * 0.82)}px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("🚗", 0, 0);
    ctx.restore();
  }

  // ---- UI events ----------------------------------------------------------

  btnConnect.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) disconnect();
    else connect();
  });

  btnReset.addEventListener("click", () => send({ type: "reset" }));

  elChatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = elChatInput.value.trim();
    if (!text) return;
    addChatMsg("user", text);
    send({ type: "user_message", text });
    elChatInput.value = "";
  });

  document.querySelectorAll(".quick-buttons button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cmd = btn.dataset.cmd;
      addChatMsg("user", cmd);
      if (cmd === "stop") send({ type: "stop" });
      else send({ type: "user_message", text: cmd });
    });
  });

  // phone status-bar clock
  const elPhoneTime = document.getElementById("phone-time");
  function updatePhoneClock() {
    if (!elPhoneTime) return;
    const d = new Date();
    let h = d.getHours();
    const m = String(d.getMinutes()).padStart(2, "0");
    h = h % 12; if (h === 0) h = 12;
    elPhoneTime.textContent = `${h}:${m}`;
  }
  updatePhoneClock();
  setInterval(updatePhoneClock, 30_000);

  // initial empty draw
  const initBg = ctx.createRadialGradient(
    canvas.width / 2, canvas.height / 2, 50,
    canvas.width / 2, canvas.height / 2, canvas.width * 0.75
  );
  initBg.addColorStop(0, "#0e1320");
  initBg.addColorStop(1, "#05070c");
  ctx.fillStyle = initBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#6b7380";
  ctx.textAlign = "center";
  ctx.font = "14px -apple-system, Inter, sans-serif";
  ctx.fillText("Click 'Connect' to start", canvas.width / 2, canvas.height / 2);
})();
