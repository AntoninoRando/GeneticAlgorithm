/* Onboarding guide — shared navigation, theme toggle and math rendering.
 *
 * To add a page: create the .html file (copy an existing one for the head and
 * layout), then add one entry to NAV below.  Order in NAV = reading order and
 * the order of the previous/next links at the bottom of every page.
 */
(function () {
  "use strict";

  var NAV = [
    { group: "Start here", items: [
      { href: "index.html",         title: "Welcome" },
      { href: "running.html",       title: "Day one: build & run" },
      { href: "demo.html",          title: "Giving a demo" }
    ]},
    { group: "Concepts", items: [
      { href: "problem.html",       title: "The scheduling problem" },
      { href: "visualization.html", title: "The scheduler in 3D" },
      { href: "gp-basics.html",     title: "Genetic programming from zero" },
      { href: "glossary.html",      title: "Glossary" }
    ]},
    { group: "The code", items: [
      { href: "architecture.html",  title: "Architecture map" },
      { href: "data.html",          title: "Data pipeline" },
      { href: "realistic-data.html", title: "Realistic data" },
      { href: "decoder.html",       title: "Decoder & fitness" },
      { href: "evolution.html",     title: "One generation, step by step" }
    ]},
    { group: "Science", items: [
      { href: "theory.html",        title: "Theory & maths" },
      { href: "results.html",       title: "Experiments & findings" },
      { href: "quirks.html",        title: "Known quirks & open questions" }
    ]},
    { group: "Reference", items: [
      { href: "where-to-start.html", title: "Where do I start if…" },
      { href: "references.html",    title: "References" },
      { href: "changelog.html",     title: "Changelog" }
    ]}
  ];

  function currentFile() {
    var p = window.location.pathname.split("/").pop();
    return p === "" ? "index.html" : decodeURIComponent(p);
  }

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // ── Theme ────────────────────────────────────────────────────────────────
  function storedTheme() {
    try { return localStorage.getItem("onboarding-theme"); } catch (e) { return null; }
  }
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(storedTheme());

  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  // ── Sidebar ──────────────────────────────────────────────────────────────
  function buildSidebar() {
    var side = document.getElementById("sidebar");
    if (!side) return;
    var here = currentFile();

    var head = el("div", { "class": "sidebar-head" });
    var brand = el("a", { "class": "brand", href: "index.html" });
    brand.appendChild(el("span", { "class": "brand-mark", "aria-hidden": "true" }, "GP"));
    var bt = el("span", { "class": "brand-text" }, "GP Scheduler");
    bt.appendChild(el("small", null, "Onboarding guide"));
    brand.appendChild(bt);
    head.appendChild(brand);
    var menu = el("button", { "class": "btn menu-toggle", type: "button", "aria-expanded": "false" }, "Menu");
    menu.addEventListener("click", function () {
      var open = side.classList.toggle("open");
      menu.setAttribute("aria-expanded", open ? "true" : "false");
    });
    head.appendChild(menu);
    side.appendChild(head);

    var wrap = el("nav", { "class": "nav-wrap", "aria-label": "Guide pages" });
    NAV.forEach(function (g) {
      wrap.appendChild(el("div", { "class": "nav-group" }, g.group));
      var ul = el("ul", { "class": "nav-list" });
      g.items.forEach(function (it) {
        var li = el("li");
        var a = el("a", { href: it.href }, it.title);
        if (it.href === here) a.setAttribute("aria-current", "page");
        li.appendChild(a);
        ul.appendChild(li);
      });
      wrap.appendChild(ul);
    });

    var tools = el("div", { "class": "sidebar-tools" });
    var themeBtn = el("button", { "class": "btn", type: "button" });
    function label() { themeBtn.textContent = isDark() ? "Light theme" : "Dark theme"; }
    label();
    themeBtn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem("onboarding-theme", next); } catch (e) { /* ignore */ }
      label();
    });
    tools.appendChild(themeBtn);
    wrap.appendChild(tools);
    side.appendChild(wrap);
  }

  // ── Previous / next ──────────────────────────────────────────────────────
  function buildPager() {
    var main = document.querySelector("main.content");
    if (!main) return;
    var flat = [];
    NAV.forEach(function (g) { g.items.forEach(function (it) { flat.push(it); }); });
    var i = -1, here = currentFile();
    for (var k = 0; k < flat.length; k++) if (flat[k].href === here) i = k;
    if (i < 0) return;
    var pager = el("nav", { "class": "pager", "aria-label": "Previous and next page" });
    if (i > 0) {
      var p = el("a", { href: flat[i - 1].href, "class": "prev" });
      p.appendChild(el("small", null, "Previous"));
      p.appendChild(document.createTextNode(flat[i - 1].title));
      pager.appendChild(p);
    }
    if (i < flat.length - 1) {
      var n = el("a", { href: flat[i + 1].href, "class": "next" });
      n.appendChild(el("small", null, "Next"));
      n.appendChild(document.createTextNode(flat[i + 1].title));
      pager.appendChild(n);
    }
    main.appendChild(pager);
  }

  // ── Math (KaTeX from cdnjs; if offline the TeX source stays readable) ───
  window.onboardingRenderMath = function () {
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  };

  function init() { buildSidebar(); buildPager(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
