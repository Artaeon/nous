package server

// webUI is the full-featured web interface with streaming, markdown rendering,
// conversation persistence, tabbed navigation, and rich chat UX.
const webUI = `<!DOCTYPE html>
<html lang="en">
<head>
<title>Nous</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/highlight.min.js"></script>
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bg4:#30363d;--fg:#e6edf3;--fg2:#8b949e;--fg3:#484f58;--accent:#58a6ff;--accent2:#1f6feb;--green:#3fb950;--yellow:#d29922;--red:#f85149;--cyan:#79c0ff;--magenta:#d2a8ff;--border:#30363d;--font:-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',system-ui,sans-serif;--mono:'SF Mono','Fira Code','JetBrains Mono',Consolas,monospace;--t:.15s ease}
*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%;overflow:hidden}
body{font-family:var(--mono);background:var(--bg);color:var(--fg);font-size:13px;-webkit-font-smoothing:antialiased}

/* Login */
.login{position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;z-index:100}.login.hidden{display:none}
.login-box{width:360px;text-align:center}
.login-box h1{font-size:28px;font-weight:300;letter-spacing:8px;color:var(--fg);margin-bottom:4px}
.login-box p{color:var(--fg2);font-size:12px;margin-bottom:32px;font-family:var(--font)}
.login-box input{width:100%;padding:12px 16px;background:var(--bg2);border:1px solid var(--border);color:var(--fg);border-radius:8px;font-family:var(--mono);font-size:14px;text-align:center;outline:none;transition:border var(--t)}
.login-box input:focus{border-color:var(--accent)}.login-box input::placeholder{color:var(--fg3)}
.login-box button{width:100%;margin-top:12px;padding:10px;background:var(--accent2);color:#fff;border:none;border-radius:8px;font-family:var(--font);font-size:14px;font-weight:600;cursor:pointer;transition:background var(--t)}
.login-box button:hover{background:var(--accent)}
.login-box .skip{margin-top:12px;font-size:11px;color:var(--fg3);cursor:pointer;font-family:var(--font)}.login-box .skip:hover{color:var(--fg2)}
.login-err{color:var(--red);font-size:12px;margin-top:8px;min-height:18px;font-family:var(--font)}

.app{display:flex;height:100vh;flex-direction:column}.app.hidden{display:none}

/* Topbar */
.topbar{display:flex;align-items:center;padding:0 16px;height:40px;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0;gap:12px}
.topbar .logo{color:var(--green);font-weight:700;font-size:13px;letter-spacing:2px}
.topbar .sep{color:var(--fg3)}
.topbar .model-info{color:var(--fg2);font-size:12px}
.topbar .tabs{display:flex;gap:2px;margin-left:24px}
.topbar .tab{padding:6px 14px;border-radius:6px;cursor:pointer;transition:all var(--t);color:var(--fg2);font-size:12px;font-family:var(--font);font-weight:500}
.topbar .tab:hover{background:var(--bg3);color:var(--fg)}
.topbar .tab.active{background:var(--bg3);color:var(--fg)}
.topbar .spacer{flex:1}
.topbar .dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px;animation:pulse 2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.topbar .status-text{font-size:11px;color:var(--fg3)}
.topbar .new-chat{padding:4px 10px;border:1px solid var(--border);border-radius:6px;cursor:pointer;color:var(--fg2);font-size:11px;font-family:var(--font);background:transparent;transition:all var(--t)}
.topbar .new-chat:hover{border-color:var(--accent);color:var(--accent)}

/* Main */
.main{flex:1;display:flex;min-height:0;overflow:hidden}
.panel{flex:1;display:none;flex-direction:column;overflow:hidden}.panel.active{display:flex}

/* Chat panel */
.chat-layout{display:flex;flex:1;min-height:0}
.chat-side{width:240px;border-right:1px solid var(--border);background:var(--bg2);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.chat-side h3{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--fg3);padding:12px 12px 8px;font-weight:600}
.chat-side-scroll{flex:1;overflow-y:auto;padding:0 8px 8px}
.chat-side-scroll::-webkit-scrollbar{width:4px}.chat-side-scroll::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:2px}
.task-item{display:flex;align-items:flex-start;gap:6px;padding:5px 8px;border-radius:6px;margin-bottom:2px;font-size:11px;cursor:pointer;transition:background var(--t)}
.task-item:hover{background:var(--bg3)}
.task-item .check{width:12px;height:12px;border:1.5px solid var(--fg3);border-radius:50%;flex-shrink:0;margin-top:2px;cursor:pointer;transition:all var(--t)}
.task-item .check:hover{border-color:var(--green);background:rgba(63,185,80,.15)}
.task-item .text{flex:1;line-height:1.4}
.task-item .due{font-size:9px;color:var(--fg3)}.task-item .due.overdue{color:var(--red)}
.job-card{padding:6px 8px;border:1px solid var(--border);border-radius:6px;margin-bottom:4px;background:var(--bg);font-size:11px}
.job-card .jmeta{display:flex;justify-content:space-between;margin-bottom:2px}
.pill{font-size:9px;font-weight:700;text-transform:uppercase;padding:1px 6px;border-radius:8px}
.pill.queued{background:#3d2e00;color:var(--yellow)}.pill.running{background:#0c2d6b;color:var(--cyan)}.pill.completed{background:#0d3117;color:var(--green)}.pill.failed,.pill.canceled{background:#3c1111;color:var(--red)}
.side-empty{color:var(--fg3);font-size:11px;text-align:center;padding:12px 0}
.sess-item{padding:5px 8px;border-radius:6px;margin-bottom:2px;font-size:11px;cursor:pointer;transition:background var(--t);color:var(--fg2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sess-item:hover{background:var(--bg3);color:var(--fg)}

.chat-main{flex:1;display:flex;flex-direction:column;min-width:0}
.output{flex:1;overflow-y:auto;padding:16px 20px}.output::-webkit-scrollbar{width:6px}.output::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}

/* Messages */
.msg{margin-bottom:16px;max-width:900px;animation:fadeUp .2s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg-user{display:flex;gap:8px;align-items:flex-start}
.msg-user .avatar{width:24px;height:24px;border-radius:50%;background:var(--accent2);color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;font-family:var(--font)}
.msg-user .content{background:var(--bg3);padding:8px 12px;border-radius:12px 12px 12px 4px;color:var(--fg);font-size:13px;line-height:1.5;max-width:80%;white-space:pre-wrap;word-break:break-word}
.msg-bot{display:flex;gap:8px;align-items:flex-start}
.msg-bot .avatar{width:24px;height:24px;border-radius:50%;background:var(--green);color:#000;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;font-family:var(--mono)}
.msg-bot .content{flex:1;min-width:0}
.msg-bot .bubble{color:var(--fg2);line-height:1.6;padding:4px 0}
.msg-bot .meta{display:flex;align-items:center;gap:8px;margin-top:4px}
.msg-bot .time{color:var(--fg3);font-size:10px}
.msg-bot .copy-btn{background:none;border:1px solid var(--border);color:var(--fg3);padding:2px 8px;border-radius:4px;font-size:10px;cursor:pointer;font-family:var(--font);transition:all var(--t)}
.msg-bot .copy-btn:hover{border-color:var(--accent);color:var(--accent)}
.msg-sys{color:var(--cyan);font-style:italic;font-size:12px;padding:4px 0 4px 32px}

/* Markdown content */
.md h1,.md h2,.md h3,.md h4,.md h5,.md h6{color:var(--fg);margin:12px 0 6px;font-family:var(--font)}
.md h1{font-size:20px;border-bottom:1px solid var(--border);padding-bottom:6px}
.md h2{font-size:17px}.md h3{font-size:15px}.md h4{font-size:13px}
.md p{margin:6px 0;color:var(--fg2)}
.md a{color:var(--accent);text-decoration:none}.md a:hover{text-decoration:underline}
.md strong{color:var(--fg);font-weight:600}
.md em{color:var(--fg2)}
.md ul,.md ol{margin:6px 0;padding-left:24px;color:var(--fg2)}
.md li{margin:3px 0;line-height:1.5}
.md blockquote{border-left:3px solid var(--accent);padding:4px 12px;margin:8px 0;color:var(--fg3);background:var(--bg2);border-radius:0 6px 6px 0}
.md hr{border:none;border-top:1px solid var(--border);margin:12px 0}
.md table{border-collapse:collapse;margin:8px 0;font-size:12px}
.md th,.md td{border:1px solid var(--border);padding:6px 10px;text-align:left}
.md th{background:var(--bg3);color:var(--fg);font-weight:600}
.md td{color:var(--fg2)}
.md code{background:var(--bg3);padding:2px 6px;border-radius:4px;font-size:12px;color:var(--cyan);font-family:var(--mono)}
.md pre{position:relative;background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:8px 0;overflow-x:auto}
.md pre code{background:none;padding:0;color:var(--fg);font-size:12px;line-height:1.5}
.md pre .copy-code{position:absolute;top:6px;right:6px;background:var(--bg3);border:1px solid var(--border);color:var(--fg3);padding:2px 8px;border-radius:4px;font-size:10px;cursor:pointer;font-family:var(--font);opacity:0;transition:opacity var(--t)}
.md pre:hover .copy-code{opacity:1}
.md pre .copy-code:hover{color:var(--accent);border-color:var(--accent)}

/* Streaming cursor */
.streaming-cursor{display:inline-block;width:2px;height:14px;background:var(--accent);animation:blink 1s step-end infinite;vertical-align:text-bottom;margin-left:2px}
@keyframes blink{50%{opacity:0}}

.thinking-dots{display:inline-flex;gap:3px;margin-left:4px}.thinking-dots span{width:5px;height:5px;border-radius:50%;background:var(--fg3);animation:bounce 1.4s ease infinite}
.thinking-dots span:nth-child(2){animation-delay:.15s}.thinking-dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:scale(.5);opacity:.3}40%{transform:scale(1);opacity:1}}

.welcome{text-align:center;padding:60px 20px;color:var(--fg3)}.welcome h2{font-size:18px;font-weight:300;letter-spacing:4px;color:var(--fg2);margin-bottom:8px;font-family:var(--font)}
.welcome p{font-size:12px;line-height:1.8;max-width:400px;margin:0 auto;font-family:var(--font)}
.welcome .cmds{margin-top:24px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.welcome .cmds span{padding:4px 10px;border:1px solid var(--border);border-radius:4px;font-size:11px;color:var(--fg2);cursor:pointer;transition:all var(--t)}
.welcome .cmds span:hover{border-color:var(--accent);color:var(--accent);background:rgba(88,166,255,.08)}

/* Input bar — textarea */
.input-bar{padding:8px 16px 12px;background:var(--bg2);border-top:1px solid var(--border);display:flex;gap:8px;align-items:flex-end;flex-shrink:0}
.input-bar textarea{flex:1;padding:8px 12px;background:var(--bg);border:1px solid var(--border);color:var(--fg);border-radius:8px;font-family:var(--mono);font-size:13px;outline:none;transition:border var(--t);resize:none;min-height:38px;max-height:120px;line-height:1.4}
.input-bar textarea:focus{border-color:var(--accent)}.input-bar textarea::placeholder{color:var(--fg3)}
.input-bar button{padding:8px 14px;border:none;border-radius:8px;font-family:var(--mono);font-size:12px;font-weight:600;cursor:pointer;transition:all var(--t);height:38px}
.input-bar .btn-go{background:var(--accent2);color:#fff}.input-bar .btn-go:hover{background:var(--accent)}
.input-bar .btn-stop{background:var(--red);color:#fff}.input-bar .btn-stop:hover{background:#da3633}
.input-bar .btn-q{background:var(--bg3);color:var(--fg2)}.input-bar .btn-q:hover{background:var(--bg4);color:var(--fg)}
.input-bar button:disabled{opacity:.3;cursor:not-allowed}

/* Content panels (dashboard, memory, tools, settings) */
.panel-scroll{flex:1;overflow-y:auto;padding:24px 32px}.panel-scroll::-webkit-scrollbar{width:6px}.panel-scroll::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}
.panel-title{font-size:18px;font-weight:600;margin-bottom:20px;font-family:var(--font);color:var(--fg)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:16px;transition:border var(--t)}
.card:hover{border-color:var(--bg4)}
.card h4{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--fg3);margin-bottom:10px;font-weight:600;font-family:var(--font)}
.card .val{font-size:28px;font-weight:700;color:var(--fg);font-family:var(--font)}
.card .sub{font-size:12px;color:var(--fg2);margin-top:4px;font-family:var(--font)}
.card-list{list-style:none}.card-list li{padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;display:flex;justify-content:space-between}
.card-list li:last-child{border:none}
.card-list .label{color:var(--fg2)}.card-list .value{color:var(--fg);font-weight:500}
.section-title{font-size:13px;font-weight:600;margin:24px 0 12px;color:var(--fg2);font-family:var(--font);text-transform:uppercase;letter-spacing:.05em}
.data-table{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:24px}
.data-table th{text-align:left;padding:8px 12px;border-bottom:2px solid var(--border);color:var(--fg3);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.06em}
.data-table td{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--fg2);font-family:var(--mono)}
.data-table tr:hover td{background:var(--bg2)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:var(--bg3);color:var(--fg2)}
.badge.green{background:#0d3117;color:var(--green)}.badge.blue{background:#0c2d6b;color:var(--cyan)}.badge.yellow{background:#3d2e00;color:var(--yellow)}
.pref-row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--border)}
.pref-row .k{color:var(--fg2);font-size:12px}.pref-row .v{color:var(--fg);font-size:12px}
.empty-msg{color:var(--fg3);text-align:center;padding:32px;font-family:var(--font)}

/* Bottom bar */
.bottombar{height:24px;background:var(--accent2);display:flex;align-items:center;padding:0 12px;font-size:11px;color:rgba(255,255,255,.85);gap:16px;flex-shrink:0;font-family:var(--font)}
.bottombar .spacer{flex:1}

@media(max-width:768px){.chat-side{display:none}.topbar .tabs{display:none}.cards{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="login" id="login"><div class="login-box">
  <h1>NOUS</h1><p>Your personal AI assistant</p>
  <input type="password" id="keyInput" placeholder="API Key" autofocus>
  <button onclick="doLogin()">Connect</button>
  <div class="skip" onclick="skipLogin()">Connect without key (local mode)</div>
  <div class="login-err" id="loginErr"></div>
</div></div>

<div class="app hidden" id="app">
<div class="topbar">
  <span class="logo">NOUS</span><span class="sep">|</span><span class="model-info" id="modelInfo">-</span>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('chat')">Chat</div>
    <div class="tab" onclick="switchTab('dashboard')">Dashboard</div>
    <div class="tab" onclick="switchTab('memory')">Memory</div>
    <div class="tab" onclick="switchTab('tools')">Tools</div>
    <div class="tab" onclick="switchTab('settings')">Settings</div>
  </div>
  <div class="spacer"></div>
  <button class="new-chat" onclick="newChat()" title="Ctrl+Shift+K">New Chat</button>
  <span class="status-text"><span class="dot"></span><span id="uptimeText">-</span></span>
</div>
<div class="main">
  <!-- CHAT TAB -->
  <div class="panel active" id="tab-chat">
    <div class="chat-layout">
      <div class="chat-side">
        <h3>Sessions</h3>
        <div class="chat-side-scroll" id="sessList"><div class="side-empty">No sessions</div></div>
        <h3>Tasks</h3>
        <div class="chat-side-scroll" id="taskList"><div class="side-empty">No tasks</div></div>
        <h3>Background Jobs</h3>
        <div class="chat-side-scroll" id="jobList"><div class="side-empty">No jobs</div></div>
      </div>
      <div class="chat-main">
        <div class="output" id="output">
          <div class="welcome"><h2>Welcome</h2><p>Fully local AI. Everything private.<br>Type a message or try a command.</p>
            <div class="cmds" id="quickCmds">
              <span onclick="runCmd('/briefing')">/briefing</span><span onclick="runCmd('/today')">/today</span>
              <span onclick="runCmd('/now')">/now</span><span onclick="runCmd('/tasks')">/tasks</span>
              <span onclick="runCmd('/compass')">/compass</span><span onclick="runCmd('/dashboard')">/dashboard</span>
              <span onclick="runCmd('/help')">/help</span>
            </div>
          </div>
        </div>
        <div class="input-bar">
          <textarea id="chatInput" placeholder="Message Nous... (Enter to send, Shift+Enter for newline)" disabled rows="1"></textarea>
          <button class="btn-go" id="btnGo" onclick="handleInput()" disabled>Send</button>
          <button class="btn-q" id="btnQ" onclick="handleQueue()" disabled title="Run in background">Queue</button>
        </div>
      </div>
    </div>
  </div>

  <!-- DASHBOARD TAB -->
  <div class="panel" id="tab-dashboard">
    <div class="panel-scroll">
      <div class="panel-title">Dashboard</div>
      <div class="cards" id="dashCards"></div>
      <div class="section-title">Training Quality</div>
      <div id="dashTraining" class="card" style="max-width:500px"></div>
    </div>
  </div>

  <!-- MEMORY TAB -->
  <div class="panel" id="tab-memory">
    <div class="panel-scroll">
      <div class="panel-title">Memory</div>
      <div class="section-title">Long-Term Memory (Persistent Facts)</div>
      <table class="data-table" id="ltmTable"><thead><tr><th>Key</th><th>Value</th><th>Category</th><th>Accessed</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Working Memory (Active Context)</div>
      <table class="data-table" id="wmTable"><thead><tr><th>Key</th><th>Value</th><th>Relevance</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Episodic Memory (Recent Interactions)</div>
      <table class="data-table" id="epTable"><thead><tr><th>Time</th><th>Input</th><th>Output</th><th>Duration</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <!-- TOOLS TAB -->
  <div class="panel" id="tab-tools">
    <div class="panel-scroll">
      <div class="panel-title">Tools &amp; Commands</div>
      <div class="section-title">Built-in Tools</div>
      <table class="data-table" id="toolsTable"><thead><tr><th>Name</th><th>Description</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Slash Commands</div>
      <div class="cards" id="cmdCards"></div>
    </div>
  </div>

  <!-- SETTINGS TAB -->
  <div class="panel" id="tab-settings">
    <div class="panel-scroll">
      <div class="panel-title">Settings</div>
      <div class="section-title">Preferences</div>
      <div class="card" id="prefsCard"></div>
      <div class="section-title">Sessions</div>
      <table class="data-table" id="sessTable"><thead><tr><th>Name</th><th>Messages</th><th>Last Updated</th><th>ID</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Conversation History</div>
      <table class="data-table" id="convTable"><thead><tr><th>Role</th><th>Content</th></tr></thead><tbody></tbody></table>
      <div class="section-title">Connection</div>
      <div class="card"><ul class="card-list">
        <li><span class="label">API Key</span><span class="value" id="settKeyStatus">-</span></li>
        <li><span class="label">Action</span><span class="value"><span style="color:var(--red);cursor:pointer" onclick="logout()">Disconnect</span></span></li>
      </ul></div>
    </div>
  </div>
</div>
<div class="bottombar">
  <span id="bbVersion">-</span><span id="bbModel">-</span><span id="bbTools">-</span>
  <span class="spacer"></span><span id="bbTasks">-</span><span id="bbJobs">-</span><span id="bbUptime">-</span>
</div>
</div>

<script>
const $=s=>document.getElementById(s);
let apiKey=localStorage.getItem('nous_api_key')||'';
let chatMessages=JSON.parse(localStorage.getItem('nous_chat_msgs')||'[]');
let streaming=false;
let abortCtrl=null;

// Markdown setup
if(typeof marked!=='undefined'){
  marked.setOptions({breaks:true,gfm:true,highlight:function(code,lang){
    if(typeof hljs!=='undefined'){
      if(lang&&hljs.getLanguage(lang))return hljs.highlight(code,{language:lang}).value;
      return hljs.highlightAuto(code).value;
    }return code;
  }});
}

function renderMd(text){
  if(typeof marked==='undefined')return'<pre style="white-space:pre-wrap">'+esc(text)+'</pre>';
  let html=marked.parse(text);
  // Add copy buttons to code blocks
  html=html.replace(/<pre><code/g,'<pre><button class="copy-code" onclick="copyCode(this)">Copy</button><code');
  return html;
}

function hdr(x){const h=Object.assign({'Content-Type':'application/json'},x||{});if(apiKey)h['Authorization']='Bearer '+apiKey;return h}
async function af(u,o){o=o||{};o.headers=hdr(o.headers);return fetch(u,o)}

// Login
$('keyInput').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
async function doLogin(){apiKey=$('keyInput').value.trim();if(!apiKey){$('loginErr').textContent='Enter API key';return}
  try{const r=await af('/api/status');if(r.status===401){$('loginErr').textContent='Invalid key';return}localStorage.setItem('nous_api_key',apiKey);enterApp()}catch(e){$('loginErr').textContent='Cannot connect'}}
function skipLogin(){apiKey='';localStorage.removeItem('nous_api_key');enterApp()}
function logout(){localStorage.removeItem('nous_api_key');location.reload()}
(async()=>{try{const r=await af('/api/health');if(r.ok){const s=await af('/api/status');if(s.ok){enterApp();return}}}catch(e){}$('login').classList.remove('hidden')})();

function enterApp(){$('login').classList.add('hidden');$('app').classList.remove('hidden');
  $('chatInput').disabled=false;$('btnGo').disabled=false;$('btnQ').disabled=false;$('chatInput').focus();
  $('settKeyStatus').textContent=apiKey?'Connected (key set)':'Local mode (no key)';
  restoreChat();
  loadAll();setInterval(loadJobs,5000);setInterval(loadTasks,10000);setInterval(loadStatus,15000)}

// Conversation persistence
function saveMessages(){
  if(chatMessages.length>200)chatMessages=chatMessages.slice(-200);
  localStorage.setItem('nous_chat_msgs',JSON.stringify(chatMessages));
}
function restoreChat(){
  if(!chatMessages.length)return;
  clearWelcome();
  chatMessages.forEach(m=>{
    if(m.role==='user')addUserMsg(m.content,false);
    else if(m.role==='assistant')addBotMsg(m.content,m.duration,false);
    else if(m.role==='system')addSys(m.content,false);
  });
  scrollBottom();
}
function newChat(){
  chatMessages=[];saveMessages();
  $('output').innerHTML='<div class="welcome"><h2>Welcome</h2><p>Fully local AI. Everything private.<br>Type a message or try a command.</p><div class="cmds" id="quickCmds"><span onclick="runCmd(\'/briefing\')">/briefing</span><span onclick="runCmd(\'/today\')">/today</span><span onclick="runCmd(\'/now\')">/now</span><span onclick="runCmd(\'/help\')">/help</span></div></div>';
  $('chatInput').focus();
}

// Tabs
function switchTab(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  $('tab-'+name).classList.add('active');
  document.querySelector('.tab[onclick*="'+name+'"]').classList.add('active');
  if(name==='dashboard')loadDashboard();if(name==='memory')loadMemory();if(name==='tools')loadTools();if(name==='settings')loadSettings();
  if(name==='chat')$('chatInput').focus();
}

function loadAll(){loadStatus();loadTasks();loadJobs();loadSessions()}

// Status
async function loadStatus(){try{const r=await af('/api/status');const s=await r.json();
  $('modelInfo').textContent=s.model+' | '+s.tool_count+' tools';$('uptimeText').textContent=s.uptime;
  $('bbVersion').textContent='v'+s.version;$('bbModel').textContent=s.model;$('bbTools').textContent=s.tool_count+' tools';$('bbUptime').textContent=s.uptime;
  $('bbJobs').textContent=(s.running_jobs+s.queued_jobs)+' jobs'}catch(e){$('modelInfo').textContent='offline'}}

// Sessions sidebar
async function loadSessions(){try{const r=await af('/api/sessions');const d=await r.json();const sl=$('sessList');sl.innerHTML='';
  if(!d.sessions||!d.sessions.length){sl.innerHTML='<div class="side-empty">No sessions</div>';return}
  d.sessions.slice(0,10).forEach(s=>{const el=document.createElement('div');el.className='sess-item';el.textContent=s.name||s.id.slice(0,12);
    el.title=s.message_count+' messages - '+new Date(s.updated_at).toLocaleString();
    sl.appendChild(el)})}catch(e){}}

// Tasks
async function loadTasks(){try{const r=await af('/api/assistant/tasks');const d=await r.json();const tl=$('taskList');tl.innerHTML='';
  if(!d.tasks||!d.tasks.length){tl.innerHTML='<div class="side-empty">No tasks</div>';$('bbTasks').textContent='0 tasks';return}
  $('bbTasks').textContent=d.tasks.length+' tasks';
  d.tasks.forEach(t=>{const el=document.createElement('div');el.className='task-item';const due=t.due_at?new Date(t.due_at):null;const ov=due&&due<new Date();
    el.innerHTML='<div class="check" onclick="event.stopPropagation();doneTask(\''+esc(t.id)+'\')"></div><span class="text">'+esc(t.title)+'</span>'+(due?'<span class="due'+(ov?' overdue':'')+'">'+fmtDate(due)+'</span>':'');
    tl.appendChild(el)})}catch(e){}}
async function doneTask(id){await af('/api/assistant/tasks/'+id+'/done',{method:'POST'});loadTasks();addSys('Task completed.')}

// Jobs
async function loadJobs(){try{const r=await af('/api/jobs');const d=await r.json();const jl=$('jobList');jl.innerHTML='';
  if(!d.jobs||!d.jobs.length){jl.innerHTML='<div class="side-empty">No jobs</div>';return}
  d.jobs.forEach(j=>{const c=document.createElement('div');c.className='job-card';
    c.innerHTML='<div class="jmeta"><span style="color:var(--fg3)">'+esc(j.id).slice(0,8)+'</span><span class="pill '+esc(j.status)+'">'+esc(j.status)+'</span></div><div class="jmsg">'+esc(j.message)+'</div>';
    jl.appendChild(c)})}catch(e){}}

// Dashboard
async function loadDashboard(){try{const r=await af('/api/dashboard');const d=await r.json();
  $('dashCards').innerHTML=[
    card('Model',d.model||'-','Active LLM'),card('Uptime',d.uptime||'-','Since last restart'),
    card('Working Memory',d.working_memory_size||0,'Active slots'),card('Long-Term Memory',d.longterm_memory_size||0,'Persistent facts'),
    card('Episodes',d.episodes_total||0,'Success rate: '+(d.success_rate*100||0).toFixed(0)+'%'),
    card('Training Pairs',d.training_pairs||0,'For fine-tuning'),
    card('Conversation',d.conversation_messages||0,'Messages this session'),card('Tasks',d.pending_tasks||0,(d.unread_notifications||0)+' unread notifications')
  ].join('');
  const qd=d.quality_distribution||{};
  $('dashTraining').innerHTML='<h4>Quality Distribution</h4><ul class="card-list">'+Object.entries(qd).map(([k,v])=>'<li><span class="label">'+esc(k)+'</span><span class="value">'+v+'</span></li>').join('')+'</ul>';
}catch(e){$('dashCards').innerHTML='<div class="empty-msg">Unable to load dashboard</div>'}}
function card(t,v,s){return'<div class="card"><h4>'+esc(t)+'</h4><div class="val">'+esc(''+v)+'</div><div class="sub">'+esc(s)+'</div></div>'}

// Memory
async function loadMemory(){
  try{const r=await af('/api/longterm');const d=await r.json();const tb=$('ltmTable').querySelector('tbody');tb.innerHTML='';
    if(d.entries)d.entries.forEach(e=>{tb.innerHTML+='<tr><td style="color:var(--cyan)">'+esc(e.key)+'</td><td>'+esc(e.value)+'</td><td><span class="badge">'+esc(e.category)+'</span></td><td>'+e.access_count+'</td></tr>'})}catch(e){}
  try{const r=await af('/api/memory');const d=await r.json();const tb=$('wmTable').querySelector('tbody');tb.innerHTML='';
    if(d.items)d.items.forEach(e=>{tb.innerHTML+='<tr><td style="color:var(--magenta)">'+esc(e.key)+'</td><td>'+esc(''+e.value)+'</td><td>'+(e.relevance||0).toFixed(2)+'</td></tr>'})}catch(e){}
  try{const r=await af('/api/episodes');const d=await r.json();const tb=$('epTable').querySelector('tbody');tb.innerHTML='';
    if(d.episodes)d.episodes.slice(0,15).forEach(e=>{const t=new Date(e.timestamp);
      tb.innerHTML+='<tr><td style="color:var(--fg3)">'+t.toLocaleTimeString()+'</td><td>'+esc((e.input||'').slice(0,60))+'</td><td style="color:var(--fg2)">'+esc((e.output||'').slice(0,80))+'</td><td>'+e.duration_ms+'ms</td></tr>'})}catch(e){}}

// Tools
async function loadTools(){
  try{const r=await af('/api/tools');const d=await r.json();const tb=$('toolsTable').querySelector('tbody');tb.innerHTML='';
    if(d.tools)d.tools.forEach(t=>{tb.innerHTML+='<tr><td style="color:var(--green);font-weight:600">'+esc(t.name)+'</td><td style="color:var(--fg2)">'+esc(t.description)+'</td></tr>'})}catch(e){}
  const cmds=[{n:'/briefing',d:'Morning briefing'},{n:'/today',d:'Tasks and notifications'},{n:'/now',d:'Next best action'},
    {n:'/compass',d:'Triage view'},{n:'/tasks',d:'List tasks'},{n:'/remind',d:'Create reminder'},{n:'/done',d:'Complete task'},
    {n:'/status',d:'Runtime status'},{n:'/dashboard',d:'System overview'},{n:'/help',d:'Command reference'},
    {n:'/knowledge',d:'Knowledge stats'},{n:'/training',d:'Training stats'},{n:'/plan',d:'Delegate a task'}];
  $('cmdCards').innerHTML=cmds.map(c=>'<div class="card" style="cursor:pointer;padding:12px" onclick="switchTab(\'chat\');runCmd(\''+c.n+'\')"><h4 style="color:var(--accent);margin-bottom:4px;font-size:13px;font-family:var(--mono)">'+c.n+'</h4><div class="sub" style="font-size:11px">'+esc(c.d)+'</div></div>').join('')}

// Settings
async function loadSettings(){
  try{const r=await af('/api/assistant/preferences');const d=await r.json();const pc=$('prefsCard');
    if(d.preferences&&d.preferences.length){pc.innerHTML=d.preferences.map(p=>'<div class="pref-row"><span class="k">'+esc(p.key)+'</span><span class="v">'+esc(p.value)+'</span></div>').join('')}
    else{pc.innerHTML='<div class="empty-msg">No preferences set. Use /pref key value</div>'}}catch(e){}
  try{const r=await af('/api/sessions');const d=await r.json();const tb=$('sessTable').querySelector('tbody');tb.innerHTML='';
    if(d.sessions)d.sessions.forEach(s=>{tb.innerHTML+='<tr><td style="color:var(--fg)">'+esc(s.name)+'</td><td>'+s.message_count+'</td><td style="color:var(--fg3)">'+new Date(s.updated_at).toLocaleString()+'</td><td style="color:var(--fg3);font-size:10px">'+esc(s.id).slice(0,12)+'</td></tr>'})}catch(e){}
  try{const r=await af('/api/conversation');const d=await r.json();const tb=$('convTable').querySelector('tbody');tb.innerHTML='';
    if(d.messages)d.messages.slice(-20).forEach(m=>{if(m.role==='system')return;
      tb.innerHTML+='<tr><td><span class="badge '+(m.role==='user'?'blue':'green')+'">'+esc(m.role)+'</span></td><td style="color:var(--fg2);max-width:700px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc((m.content||'').slice(0,200))+'</td></tr>'})}catch(e){}}

// Textarea auto-resize
$('chatInput').addEventListener('input',function(){this.style.height='auto';this.style.height=Math.min(this.scrollHeight,120)+'px'});
$('chatInput').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleInput()}
});

function runCmd(cmd){$('chatInput').value=cmd;handleInput()}

// Chat message rendering
function addUserMsg(text,save=true){
  clearWelcome();
  const d=document.createElement('div');d.className='msg msg-user';
  d.innerHTML='<div class="avatar">U</div><div class="content">'+esc(text)+'</div>';
  $('output').appendChild(d);scrollBottom();
  if(save){chatMessages.push({role:'user',content:text,ts:Date.now()});saveMessages()}
}

function addBotMsg(text,duration,save=true){
  const d=document.createElement('div');d.className='msg msg-bot';
  const durStr=duration?duration+'ms':'';
  d.innerHTML='<div class="avatar">N</div><div class="content"><div class="bubble md">'+renderMd(text)+'</div><div class="meta"><span class="time">'+durStr+'</span><button class="copy-btn" onclick="copyMsg(this)">Copy</button></div></div>';
  $('output').appendChild(d);scrollBottom();
  if(save){chatMessages.push({role:'assistant',content:text,duration:duration,ts:Date.now()});saveMessages()}
}

function addBotStreaming(){
  const d=document.createElement('div');d.className='msg msg-bot';d.id='streaming-msg';
  d.innerHTML='<div class="avatar">N</div><div class="content"><div class="bubble md"><span class="thinking-dots"><span></span><span></span><span></span></span></div><div class="meta"><span class="time"></span><button class="copy-btn" onclick="copyMsg(this)">Copy</button></div></div>';
  $('output').appendChild(d);scrollBottom();
  return d;
}

function addSys(text,save=true){
  const d=document.createElement('div');d.className='msg-sys';d.textContent=text;
  $('output').appendChild(d);scrollBottom();
  if(save){chatMessages.push({role:'system',content:text,ts:Date.now()});saveMessages()}
}

function scrollBottom(){const o=$('output');const atBottom=o.scrollHeight-o.scrollTop-o.clientHeight<100;if(atBottom)o.scrollTop=o.scrollHeight}

// Streaming chat
async function handleInput(){
  const msg=$('chatInput').value.trim();if(!msg)return;
  $('chatInput').value='';$('chatInput').style.height='auto';
  clearWelcome();

  if(msg.startsWith('/')){addUserMsg(msg);if(await handleSlash(msg))return;return}

  addUserMsg(msg);
  streaming=true;showStopBtn();

  abortCtrl=new AbortController();
  const el=addBotStreaming();
  const bubble=el.querySelector('.bubble');
  const timEl=el.querySelector('.time');
  let fullText='';
  let renderTimer=null;

  function doRender(){
    bubble.innerHTML=renderMd(fullText)+'<span class="streaming-cursor"></span>';
    scrollBottom();
  }

  try{
    const r=await fetch('/api/chat/stream',{
      method:'POST',
      headers:hdr(),
      body:JSON.stringify({message:msg}),
      signal:abortCtrl.signal
    });

    const reader=r.body.getReader();
    const decoder=new TextDecoder();
    let buf='';

    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=decoder.decode(value,{stream:true});

      let nl;
      while((nl=buf.indexOf('\n'))!==-1){
        const line=buf.slice(0,nl).trim();buf=buf.slice(nl+1);
        if(!line.startsWith('data:'))continue;
        try{
          const ev=JSON.parse(line.slice(5).trim());
          if(ev.d){
            // Done
            if(ev.ms)timEl.textContent=ev.ms+'ms';
            bubble.innerHTML=renderMd(fullText);
            chatMessages.push({role:'assistant',content:fullText,duration:ev.ms||0,ts:Date.now()});
            saveMessages();
          }else{
            fullText+=ev.t;
            // Debounced render
            if(!renderTimer)renderTimer=setTimeout(()=>{renderTimer=null;doRender()},50);
          }
        }catch(pe){}
      }
    }
    // Final render
    if(renderTimer){clearTimeout(renderTimer);renderTimer=null}
    bubble.innerHTML=renderMd(fullText);
  }catch(e){
    if(e.name==='AbortError'){
      bubble.innerHTML=renderMd(fullText)+'<div style="color:var(--yellow);font-size:11px;margin-top:4px">(stopped)</div>';
      chatMessages.push({role:'assistant',content:fullText+'\\n\\n(stopped)',ts:Date.now()});saveMessages();
    }else{
      bubble.innerHTML='<span style="color:var(--red)">Error: '+esc(e.message)+'</span>';
    }
  }
  el.removeAttribute('id');
  streaming=false;showSendBtn();
}

function stopStreaming(){if(abortCtrl)abortCtrl.abort()}

function showStopBtn(){
  $('btnGo').style.display='none';
  const stop=document.createElement('button');stop.className='btn-stop';stop.id='btnStop';stop.textContent='Stop';stop.onclick=stopStreaming;
  $('btnGo').parentNode.insertBefore(stop,$('btnGo'));
  $('chatInput').disabled=true;$('btnQ').disabled=true;
}
function showSendBtn(){
  const stop=$('btnStop');if(stop)stop.remove();
  $('btnGo').style.display='';
  $('chatInput').disabled=false;$('btnQ').disabled=false;$('chatInput').focus();
}

async function handleQueue(){const msg=$('chatInput').value.trim();if(!msg)return;$('chatInput').value='';$('chatInput').style.height='auto';clearWelcome();addUserMsg(msg);
  try{await af('/api/jobs',{method:'POST',body:JSON.stringify({message:msg})});addSys('Queued for background execution.');loadJobs()}catch(e){addSys('Error: '+e.message)}$('chatInput').focus()}

async function handleSlash(cmd){const p=cmd.split(/\s+/);const c=p[0].toLowerCase();
  try{
    if(c==='/tasks'){const r=await af('/api/assistant/tasks');const d=await r.json();
      if(!d.tasks||!d.tasks.length){addSys('No pending tasks.');return true}
      addBotMsg(d.tasks.map(t=>(t.status==='done'?'- [x]':'- [ ]')+' '+t.title+(t.due_at?' ('+fmtDate(new Date(t.due_at))+')':'')+' **#'+t.id+'**').join('\n'));return true}
    if(c==='/today'){const r=await af('/api/assistant/today');const d=await r.json();let o='';
      if(d.notifications&&d.notifications.length)o+='**Notifications:**\n'+d.notifications.map(n=>'- '+n.message).join('\n')+'\n\n';
      if(d.today&&d.today.length)o+='**Today:**\n'+d.today.map(t=>'- '+t.title).join('\n')+'\n\n';
      if(d.upcoming&&d.upcoming.length)o+='**Upcoming:**\n'+d.upcoming.map(t=>'- '+t.title+(t.due_at?' — '+fmtDate(new Date(t.due_at)):'')).join('\n');
      addBotMsg(o||'All clear.');return true}
    if(c==='/status'){const r=await af('/api/status');const d=await r.json();
      addBotMsg('| Field | Value |\n|-------|-------|\n| Version | '+d.version+' |\n| Model | '+d.model+' |\n| Tools | '+d.tool_count+' |\n| Uptime | '+d.uptime+' |\n| Percepts | '+d.percepts+' |\n| Jobs | '+d.running_jobs+' running, '+d.queued_jobs+' queued |');return true}
    if(c==='/prefs'){const r=await af('/api/assistant/preferences');const d=await r.json();
      if(!d.preferences||!d.preferences.length){addSys('No preferences set.');return true}
      addBotMsg(d.preferences.map(p=>'- **'+p.key+'**: '+p.value).join('\n'));return true}
    if(c==='/pref'&&p.length>=3){await af('/api/assistant/preferences',{method:'POST',body:JSON.stringify({key:p[1],value:p.slice(2).join(' ')})});addSys('Saved: '+p[1]);return true}
    if(c==='/routines'){const r=await af('/api/assistant/routines');const d=await r.json();
      if(!d.routines||!d.routines.length){addSys('No routines.');return true}
      addBotMsg(d.routines.map(r=>'- '+(r.enabled?'**ON**':'OFF')+' '+r.title+' ('+r.schedule+' @ '+r.time_of_day+')').join('\n'));return true}
    if(c==='/remind'&&p.length>=2){await af('/api/assistant/tasks',{method:'POST',body:JSON.stringify({title:p.slice(1).join(' ')})});addSys('Reminder created.');loadTasks();return true}
    if(c==='/done'&&p[1]){await af('/api/assistant/tasks/'+p[1]+'/done',{method:'POST'});addSys('Done.');loadTasks();return true}
    if(c==='/clear'){newChat();return true}
  }catch(e){addSys('Error: '+e.message);return true}
  return false}

// Utilities
function clearWelcome(){const w=document.querySelector('.welcome');if(w)w.remove()}
function copyMsg(btn){const bubble=btn.closest('.content').querySelector('.bubble');
  const text=bubble.innerText||bubble.textContent;navigator.clipboard.writeText(text);
  btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',1500)}
function copyCode(btn){const code=btn.nextElementSibling;navigator.clipboard.writeText(code.textContent);
  btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',1500)}
function esc(t){return(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtDate(d){const now=new Date();const diff=d-now;const days=Math.ceil(diff/864e5);if(days===0)return'today';if(days===1)return'tomorrow';if(days===-1)return'yesterday';if(days<-1)return Math.abs(days)+'d ago';if(days<7)return'in '+days+'d';return d.toLocaleDateString([],{month:'short',day:'numeric'})}

// Keyboard shortcuts
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.shiftKey&&e.key==='K'){e.preventDefault();newChat()}
  if(e.key==='Escape'&&streaming)stopStreaming();
  if(e.key==='/'&&document.activeElement!==$('chatInput')){e.preventDefault();$('chatInput').focus()}
});
</script>
</body>
</html>`
