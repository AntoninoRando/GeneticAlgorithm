/* Onboarding guide — formula parser shared by the interactive widgets
 * (decoder playground in decoder.html, 3D view in visualization.html).
 *
 * OnboardingFormula.parse("-(job.executionTime + job.transferTime)")
 * returns a function (job, sat) -> number, or throws an Error with a message
 * a reader can act on.  Operators and protected division follow
 * include/scheduler/expr_tree.h; only a subset of the 18 terminals of
 * registry.h is offered (see TERMS).
 */
(function () {
  "use strict";

  var TERMS = {
    "job.arrivalTime":     function (j) { return j.arrivalTime; },
    "job.executionTime":   function (j) { return j.executionTime; },
    "job.transferTime":    function (j) { return j.transferTime; },
    "job.imageSize":       function (j) { return j.imageSize; },
    "sat.remainingEnergy": function (j, s) { return s.remainingEnergy; },
    "sat.completedTasks":  function (j, s) { return s.completedTasks; },
    "sat.bandwidth":       function (j, s) { return s.bandwidth; },
    "sat.cpuCapacity":     function (j, s) { return s.cpuCapacity; }
  };

  // Protected division: 0 when the divisor is (almost) 0.
  function pdiv(a, b) { return Math.abs(b) < 1e-9 ? 0 : a / b; }

  var FUNCS = {
    MAX: [2, function (a, b) { return Math.max(a, b); }],
    MIN: [2, function (a, b) { return Math.min(a, b); }],
    ABS: [1, function (a) { return Math.abs(a); }],
    NEG: [1, function (a) { return -a; }],
    INV: [1, function (a) { return Math.abs(a) < 1e-9 ? 0 : 1 / a; }]
  };

  // Tiny recursive-descent parser → closure (job, sat) => number
  function parse(src) {
    var toks = [], re = /\s*(\/\?|[A-Za-z_][\w.]*|\d+(?:\.\d+)?|[-+*\/(),])/y, m, pos = 0;
    src = String(src).trim();
    if (!src) throw new Error("The formula is empty");
    while (pos < src.length) {
      re.lastIndex = pos; m = re.exec(src);
      if (!m) throw new Error("Unexpected character at position " + (pos + 1) + ": '" + src[pos] + "'");
      toks.push(m[1]); pos = re.lastIndex;
      while (pos < src.length && /\s/.test(src[pos])) pos++;
    }
    var i = 0;
    function peek() { return toks[i]; }
    function eat(t) { if (toks[i] !== t) throw new Error("Expected '" + t + "' but found " + (toks[i] ? "'" + toks[i] + "'" : "end of formula")); i++; }
    function expr() {
      var f = term();
      while (peek() === "+" || peek() === "-") {
        var op = toks[i++], g = term();
        f = (function (a, b, o) { return o === "+" ? function (j, s) { return a(j, s) + b(j, s); } : function (j, s) { return a(j, s) - b(j, s); }; })(f, g, op);
      }
      return f;
    }
    function term() {
      var f = unary();
      while (peek() === "*" || peek() === "/" || peek() === "/?") {
        var op = toks[i++], g = unary();
        f = (function (a, b, o) { return o === "*" ? function (j, s) { return a(j, s) * b(j, s); } : function (j, s) { return pdiv(a(j, s), b(j, s)); }; })(f, g, op);
      }
      return f;
    }
    function unary() {
      if (peek() === "-") { i++; var a = unary(); return function (j, s) { return -a(j, s); }; }
      return primary();
    }
    function primary() {
      var t = toks[i];
      if (t === undefined) throw new Error("Formula ends too early");
      if (t === "(") { i++; var e = expr(); eat(")"); return e; }
      if (/^\d/.test(t)) { i++; var v = parseFloat(t); return function () { return v; }; }
      var up = t.toUpperCase();
      if (FUNCS[up] && toks[i + 1] === "(") {
        i += 2; var args = [expr()];
        while (peek() === ",") { i++; args.push(expr()); }
        eat(")");
        if (args.length !== FUNCS[up][0]) throw new Error(up + " takes " + FUNCS[up][0] + " argument(s)");
        var fn = FUNCS[up][1];
        return args.length === 1 ? function (j, s) { return fn(args[0](j, s)); }
                                 : function (j, s) { return fn(args[0](j, s), args[1](j, s)); };
      }
      if (TERMS[t]) { i++; var acc = TERMS[t]; return function (j, s) { return acc(j, s); }; }
      throw new Error("Unknown name '" + t + "'");
    }
    var f = expr();
    if (i < toks.length) throw new Error("Unexpected '" + toks[i] + "'");
    return f;
  }

  window.OnboardingFormula = { parse: parse, terminals: Object.keys(TERMS) };
})();
