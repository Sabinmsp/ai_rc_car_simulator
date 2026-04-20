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
  const elLogs       = $("logs");

  const dbg = {
    cmd:     $("dbg-cmd"),
    pos:     $("dbg-pos"),
    dir:     $("dbg-dir"),
    visible: $("dbg-visible"),
    action:  $("dbg-action"),
    llm:     $("dbg-llm"),
  };

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
        updateDebugFromWorld();
        break;
      case "chat":
        addChatMsg(msg.role, msg.text);
        break;
      case "log":
        addLogRow(msg.channel, msg.text, msg.data);
        if (msg.channel === "system" && msg.data && typeof msg.data === "object" && "next_action" in msg.data) {
          dbg.llm.textContent = JSON.stringify(msg.data, null, 2);
        }
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

  function addLogRow(channel, text, data) {
    const map = {
      "A->B":   "A2B",
      "B->C":   "B2C",
      "C->B":   "C2B",
      "B->A":   "B2A",
      "system": "SYS",
    };
    const cls = map[channel] || "SYS";
    const time = new Date().toLocaleTimeString();
    const row = document.createElement("div");
    row.className = "log-row";
    const dataStr = data && Object.keys(data).length ? ` ${JSON.stringify(data)}` : "";
    row.innerHTML = `
      <span class="time">${time}</span>
      <span class="ch ${cls}">${channel}</span>
      <span class="text">${escapeHtml(text)}<span class="data">${escapeHtml(dataStr)}</span></span>
    `;
    elLogs.appendChild(row);
    elLogs.scrollTop = elLogs.scrollHeight;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ---- Debug panel --------------------------------------------------------

  function updateDebugFromWorld() {
    if (!world) return;
    dbg.pos.textContent = `(${world.car.x}, ${world.car.y})`;
    dbg.dir.textContent = world.car.dir;
    dbg.action.textContent = world.last_action || "-";
    // visible objects come from C->B logs; here we approximate from FOV cells
    const fovSet = new Set(world.fov_cells.map(([x, y]) => `${x},${y}`));
    const visible = world.objects.filter(o => fovSet.has(`${o.x},${o.y}`)).map(o => o.name);
    dbg.visible.textContent = visible.length ? visible.join(", ") : "(none)";
  }

  // ---- Room canvas --------------------------------------------------------

  function drawWorld() {
    if (!world) return;
    const W = canvas.width, H = canvas.height;
    const N = world.grid_size;
    const cell = Math.floor(Math.min(W, H) / N);

    ctx.clearRect(0, 0, W, H);

    // grid background
    ctx.fillStyle = "#0a0c10";
    ctx.fillRect(0, 0, W, H);

    // FOV cells
    ctx.fillStyle = "rgba(250, 204, 21, 0.20)";
    for (const [x, y] of world.fov_cells) {
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }

    // walls (drawn before grid lines so the lines outline them subtly)
    ctx.fillStyle = "#475569";
    const walls = world.walls || [];
    for (const [x, y] of walls) {
      ctx.fillRect(x * cell, y * cell, cell, cell);
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 2;
      ctx.strokeRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
    }

    // grid lines
    ctx.strokeStyle = "#1f2530";
    ctx.lineWidth = 1;
    for (let i = 0; i <= N; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, N * cell); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(N * cell, i * cell); ctx.stroke();
    }

    // objects
    const colors = {
      football: "#f97316",
      key:      "#facc15",
      chair:    "#a78bfa",
      table:    "#94a3b8",
    };
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const o of world.objects) {
      const cx = o.x * cell + cell / 2;
      const cy = o.y * cell + cell / 2;
      ctx.fillStyle = colors[o.name] || "#fff";
      ctx.beginPath();
      ctx.arc(cx, cy, cell * 0.35, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#0a0c10";
      ctx.fillText(o.name[0].toUpperCase(), cx, cy);
    }

    // car
    drawCar(ctx, world.car, cell);
  }

  function drawCar(ctx, car, cell) {
    const cx = car.x * cell + cell / 2;
    const cy = car.y * cell + cell / 2;
    const r = cell * 0.4;

    ctx.save();
    ctx.translate(cx, cy);
    const rot = { N: -Math.PI / 2, E: 0, S: Math.PI / 2, W: Math.PI }[car.dir] || 0;
    ctx.rotate(rot);

    // body
    ctx.fillStyle = "#60a5fa";
    ctx.fillRect(-r, -r * 0.6, r * 2, r * 1.2);

    // nose triangle (points to "forward" = +x in local frame)
    ctx.fillStyle = "#bfdbfe";
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(r * 0.4, -r * 0.6);
    ctx.lineTo(r * 0.4,  r * 0.6);
    ctx.closePath();
    ctx.fill();

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
    dbg.cmd.textContent = text;
    send({ type: "user_message", text });
    elChatInput.value = "";
  });

  document.querySelectorAll(".quick-buttons button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cmd = btn.dataset.cmd;
      addChatMsg("user", cmd);
      dbg.cmd.textContent = cmd;
      if (cmd === "stop") send({ type: "stop" });
      else send({ type: "user_message", text: cmd });
    });
  });

  // initial empty draw
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "center";
  ctx.font = "14px sans-serif";
  ctx.fillText("Click 'Connect' to start", canvas.width / 2, canvas.height / 2);
})();
