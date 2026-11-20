/* Onboarding guide — the 3D view (visualization.html).
 *
 * The file has two parts:
 *
 *   1. MODEL  A mirror of the selection and commit blocks of
 *             detail::greedyDecode (include/scheduler/fitness_evaluator.h).
 *             No drawing code.  Viz3D.selfTest() replays the worked examples
 *             of decoder.html and checks the result.
 *   2. SCENE  The three.js picture: Earth, satellites, links, moving tasks.
 *             Illustrative only.  Nothing in part 2 decides where a task goes.
 *
 * If the C++ decoder changes, change part 1, the self-test and the page text.
 * Needs assets/formula.js (the formula parser) and three.js r128 (global THREE).
 */
(function () {
  "use strict";

  // ═══════════════════════════════════════════════════════════════════════
  // 1. MODEL
  // ═══════════════════════════════════════════════════════════════════════

  // d_j = e_j + t_j.  The same on every satellite, as in the C++ code.
  function duration(job) { return job.executionTime + job.transferTime; }

  // Why (job, sat) cannot be committed now, or null when it can.
  // Same two tests, in the same order, as greedyDecode.
  function infeasibleReason(job, sat) {
    var d = duration(job);
    if (sat.freeAt + d > sat.operationalUntil) return "sunset";   // f_s + d_j <= omega_s
    if (sat.rate * d > sat.remainingEnergy) return "energy";      // rho_s d_j <= E_s
    return null;
  }

  // greedyDecode starts every satellite at "free at the snapshot minute".
  // Here the world lives on between rounds, so an idle satellite is moved up
  // to the current minute instead.
  function advanceClock(sats, tau) {
    for (var s = 0; s < sats.length; s++) if (sats[s].freeAt < tau) sats[s].freeAt = tau;
  }

  // One round of the decoder: the feasible (job, satellite) pair with the
  // highest score.  `waiting` must be in arrival order (= job index order).
  // Returns null when no pair can be committed.
  function pickBest(waiting, sats, f, earliestFree) {
    var bestScore = -Infinity, bj = -1, bs = -1, bestStart = 0;
    for (var j = 0; j < waiting.length; j++) {
      for (var s = 0; s < sats.length; s++) {
        if (infeasibleReason(waiting[j], sats[s])) continue;
        var sc = f(waiting[j], sats[s]);
        var start = sats[s].freeAt;
        if (!earliestFree) {
          // Default: strictly higher score wins, so ties go to the first pair
          // of the scan (lowest job index, then lowest satellite index).
          if (sc > bestScore) { bestScore = sc; bj = j; bs = s; }
        } else {
          // --earliest-free-tie-break: (score, -start, -job, -satellite),
          // NaN counts as the lowest score, first feasible pair is accepted.
          var value = (sc !== sc) ? -Infinity : sc;
          if (bj === -1 || value > bestScore || (value === bestScore && start < bestStart)) {
            bestScore = value; bj = j; bs = s; bestStart = start;
          }
        }
      }
    }
    return bj < 0 ? null : { job: waiting[bj], sat: sats[bs], score: bestScore };
  }

  // Commit block of greedyDecode.
  function commit(job, sat) {
    var d = duration(job), cost = sat.rate * d;
    job.sat = sat;
    job.start = sat.freeAt;
    job.end = job.start + d;
    sat.freeAt = job.end;
    sat.completedTasks += 1;
    sat.spent += cost;
    sat.remainingEnergy -= cost;
  }

  function makeSat(index, energy, sunset, bandwidth, cpu) {
    return { index: index, name: "S" + index, remainingEnergy: energy, energy0: energy,
             operationalUntil: sunset, bandwidth: bandwidth, cpuCapacity: cpu,
             completedTasks: 0, freeAt: 0, spent: 0, rate: 1 };
  }

  // Decode a fixed batch to the end (used by the self-test).
  function decodeAll(jobs, sats, f, tau, earliestFree) {
    var waiting = jobs.slice(), order = [];
    advanceClock(sats, tau);
    for (;;) {
      var best = pickBest(waiting, sats, f, earliestFree);
      if (!best) break;
      commit(best.job, best.sat);
      waiting.splice(waiting.indexOf(best.job), 1);
      order.push(best.job.name + ">" + best.sat.name + "@" + best.job.end);
    }
    return order.join(" ");
  }

  function selfTest() {
    var parse = window.OnboardingFormula.parse, checks = [];
    function check(name, got, want) { checks.push({ name: name, ok: got === want, got: got, want: want }); }
    function J(name, a, e, t) { return { name: name, arrivalTime: a, executionTime: e, transferTime: t, imageSize: 10 }; }

    // decoder.html, section 2: shortest job first, tau = 10.
    var jobs = [J("J0", 0, 10, 2), J("J1", 3, 5, 1), J("J2", 7, 20, 4), J("J3", 9, 8, 2), J("J4", 5, 15, 3)];
    var sats = [makeSat(0, 20, 40, 12, 3.1), makeSat(1, 30, 35, 9, 4.2)];
    check("worked example (SJF)",
          decodeAll(jobs, sats, parse("-(job.executionTime + job.transferTime)"), 10, false),
          "J1>S0@16 J3>S0@26 J0>S1@22");
    check("worked example, energy spent", sats[0].spent + sats[1].spent, 28);

    // decoder.html, "Ties": four 10-minute jobs, two identical satellites.
    function four() { return [J("A", 0, 10, 0), J("B", 0, 10, 0), J("C", 0, 10, 0), J("D", 0, 10, 0)]; }
    function two() { return [makeSat(0, 1e6, 1e6, 1, 1), makeSat(1, 1e6, 1e6, 1, 1)]; }
    check("ties, default (index order)",
          decodeAll(four(), two(), parse("job.executionTime"), 0, false),
          "A>S0@10 B>S0@20 C>S0@30 D>S0@40");
    check("ties, earliest free",
          decodeAll(four(), two(), parse("job.executionTime"), 0, true),
          "A>S0@10 B>S1@10 C>S0@20 D>S1@20");

    // decoder.html, "It is not always better": 30 energy units, jobs 10, 10, 30.
    function three() { return [J("A", 0, 10, 0), J("B", 0, 10, 0), J("C", 0, 30, 0)]; }
    function tight() { return [makeSat(0, 30, 1e6, 1, 1), makeSat(1, 30, 1e6, 1, 1)]; }
    check("tight batteries, default",
          decodeAll(three(), tight(), parse("-job.executionTime"), 0, false),
          "A>S0@10 B>S0@20 C>S1@30");
    check("tight batteries, earliest free",
          decodeAll(three(), tight(), parse("-job.executionTime"), 0, true),
          "A>S0@10 B>S1@10");

    return { ok: checks.every(function (c) { return c.ok; }), checks: checks };
  }

  // ── A small world to play in ───────────────────────────────────────────
  // Seeded generator (mulberry32) so that "world 1" is the same for everyone.
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function round1(x) { return Math.round(x * 10) / 10; }

  var N_PLANES = 3, PER_PLANE = 5, N_SAT = N_PLANES * PER_PLANE;

  // "roomy" is larger than our data so that a world lasts longer on screen;
  // "tight" uses the ranges of the CSV worlds (see problem.html).
  var WORLD_PRESETS = {
    roomy: { energy: [60, 120], sunset: [60, 140] },
    tight: { energy: [14, 26],  sunset: [300, 600] }
  };

  function newWorld(seed, presetName) {
    var rnd = mulberry32(seed), p = WORLD_PRESETS[presetName] || WORLD_PRESETS.roomy, sats = [];
    function between(r) { return Math.round(r[0] + (r[1] - r[0]) * rnd()); }
    for (var i = 0; i < N_SAT; i++)
      sats.push(makeSat(i, between(p.energy), between(p.sunset), round1(6 + 8 * rnd()), round1(2 + 3 * rnd())));
    return { seed: seed, preset: presetName, rnd: rnd, sats: sats, jobs: [], clock: 0, tau: 0, nextId: 0,
             created: 0, scheduled: 0, rejected: 0, responseSum: 0 };
  }

  // Ranges of problem.html: execution 5-30 min, transfer 1-8 min, image 8-24.
  // Arrivals: on average 10 per minute, as in a real world (100 jobs in 10 min).
  function newJob(world) {
    var r = world.rnd;
    world.clock += -Math.log(1 - r()) * 0.1;
    world.tau = Math.floor(world.clock);
    var id = world.nextId++;
    return { id: id, name: "T" + id, arrivalTime: world.tau,
             executionTime: 5 + Math.floor(r() * 26), transferTime: 1 + Math.floor(r() * 8),
             imageSize: round1(8 + 16 * r()), state: "uplink", sat: null, start: 0, end: 0, ap: -1, slot: 0, vis: null };
  }

  // ── Geometry helpers without three.js (plain x, y, z) ──────────────────
  var TWO_PI = Math.PI * 2;
  var ORBIT_R = 1.55;                  // Earth radius = 1.  Real LEO would be about 1.09: too close to see.
  var INCLINATION = 55 * Math.PI / 180;
  var PERIOD = 90;                     // seconds of wall time for one orbit
  var GS_LAT = 42, GS_LON = 12.5;      // the ground user

  function latLon(lat, lon, r) {
    var a = lat * Math.PI / 180, b = lon * Math.PI / 180;
    return { x: r * Math.cos(a) * Math.cos(b), y: r * Math.sin(a), z: -r * Math.cos(a) * Math.sin(b) };
  }
  function planeAngle(p, t) { return (p / N_PLANES) * TWO_PI - (t / (PERIOD * 16)) * TWO_PI; }
  function satPosition(i, t, out) {
    var p = Math.floor(i / PER_PLANE), k = i % PER_PLANE;
    var u = (k / PER_PLANE) * TWO_PI + p * 0.42 + (t / PERIOD) * TWO_PI;
    var raan = planeAngle(p, t);
    var x1 = ORBIT_R * Math.cos(u), w = ORBIT_R * Math.sin(u);
    var y1 = w * Math.sin(INCLINATION), z1 = w * Math.cos(INCLINATION);
    out.x = x1 * Math.cos(raan) + z1 * Math.sin(raan);
    out.y = y1;
    out.z = -x1 * Math.sin(raan) + z1 * Math.cos(raan);
    return out;
  }
  function dist(a, b) { var x = a.x - b.x, y = a.y - b.y, z = a.z - b.z; return Math.sqrt(x * x + y * y + z * z); }
  // True when the straight line a-b stays clear of the Earth (radius 1) by a margin.
  function segmentClear(a, b, margin) {
    var abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
    var t = -(a.x * abx + a.y * aby + a.z * abz) / (abx * abx + aby * aby + abz * abz);
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    var x = a.x + t * abx, y = a.y + t * aby, z = a.z + t * abz;
    return x * x + y * y + z * z >= margin * margin;
  }
  // Inter-satellite links: every satellite links to its 3 nearest neighbours in line of sight.
  function buildLinks(pos) {
    var n = pos.length, los = [], adj = [], pairs = [], seen = {}, i, j;
    for (i = 0; i < n; i++) { los.push([]); adj.push([]); }
    for (i = 0; i < n; i++) for (j = i + 1; j < n; j++)
      if (segmentClear(pos[i], pos[j], 1.1)) { los[i].push(j); los[j].push(i); }
    for (i = 0; i < n; i++) {
      var near = los[i].slice().sort(function (a, b) { return dist(pos[i], pos[a]) - dist(pos[i], pos[b]); }).slice(0, 3);
      for (j = 0; j < near.length; j++) {
        var a = Math.min(i, near[j]), b = Math.max(i, near[j]), key = a * 100 + b;
        if (!seen[key]) { seen[key] = true; pairs.push([a, b]); adj[a].push(b); adj[b].push(a); }
      }
    }
    return { los: los, adj: adj, pairs: pairs };
  }
  function bfs(adj, from, to) {
    var prev = [], queue = [from], k = 0, i;
    for (i = 0; i < adj.length; i++) prev.push(-2);
    prev[from] = -1;
    while (k < queue.length) {
      var u = queue[k++];
      if (u === to) break;
      for (i = 0; i < adj[u].length; i++) if (prev[adj[u][i]] === -2) { prev[adj[u][i]] = u; queue.push(adj[u][i]); }
    }
    if (prev[to] === -2) return null;
    var path = [];
    for (var v = to; v !== -1; v = prev[v]) path.unshift(v);
    return path;
  }
  // Fewest hops over the drawn links; any line of sight as a fallback.
  function route(links, from, to) {
    return from === to ? [from] : (bfs(links.adj, from, to) || bfs(links.los, from, to) || [from, to]);
  }
  // The access point: nearest satellite above the ground user's horizon.
  function pickAccessPoint(pos, gs) {
    var best = -1, bd = Infinity, any = 0, ad = Infinity;
    for (var i = 0; i < pos.length; i++) {
      var d = dist(pos[i], gs);
      var up = ((pos[i].x - gs.x) * gs.x + (pos[i].y - gs.y) * gs.y + (pos[i].z - gs.z) * gs.z) / d;  // sine of the elevation
      if (up > 0.1 && d < bd) { bd = d; best = i; }
      if (d < ad) { ad = d; any = i; }
    }
    return best >= 0 ? best : any;
  }

  window.Viz3D = {
    selfTest: selfTest,
    _model: { pickBest: pickBest, commit: commit, infeasibleReason: infeasibleReason, newWorld: newWorld, newJob: newJob },
    _geom: { satPosition: satPosition, buildLinks: buildLinks, route: route, pickAccessPoint: pickAccessPoint, latLon: latLon }
  };

  if (typeof document === "undefined") return;   // loaded outside a browser (tests)

  // ═══════════════════════════════════════════════════════════════════════
  // 2. SCENE
  // ═══════════════════════════════════════════════════════════════════════

  // Coarse coastlines as [lon, lat, lon, lat, ...].  Hand-simplified: good
  // enough to recognise the continents, not a map.
  var LAND = [
    // North and Central America
    [-168,66,-162,67,-166,68.5,-156,71,-141,70,-128,70,-115,68.5,-108,68,-95,68.5,-90,64.5,-94.5,59,-92.5,57,-85,55.3,-82.3,55,-82,52.5,-79.5,51.5,-79,54.5,-77,57,-78,59,-77.5,62.3,-73,62,-70,61,-66,59,-64,60.3,-61.5,56.5,-56,52,-60,50,-66,50.2,-65,48.5,-61,46,-64,44.5,-66,44.8,-70,43.5,-70.5,41.8,-74,40.5,-75.5,37.8,-76,35.2,-81,31.5,-80,27,-80.3,25.3,-82,26.5,-82.7,29,-85,29.8,-89,30.2,-90,29.2,-94,29.5,-97.2,27.5,-97.7,22,-95,18.8,-91,18.6,-90.3,21,-87,21.5,-88,16,-84,15.6,-83.3,11,-80,9,-77.5,8.5,-78.5,7.5,-80,7.3,-83,8.4,-85.8,11,-87.7,13,-92,14.5,-96,15.8,-102,18,-105.5,20,-106,23,-109,25.5,-112.5,29.5,-114.6,31.5,-114.5,30,-112,26.5,-109.8,23.2,-112,25,-114.5,28,-116,30.5,-117.2,32.6,-120.6,34.5,-122.5,37.6,-124.2,40.4,-124,46.5,-123,48.4,-128,51,-131,54.6,-135,58,-141,60,-148,60.6,-152,59,-158,56.6,-163,55,-160,58.6,-165,61,-164,64],
    // South America
    [-77.5,8.5,-72,12,-67,10.5,-62,10.5,-57,6,-52,4.5,-50,0.5,-44,-2.5,-38,-4,-35,-7,-37,-12,-39,-17,-41,-22,-45,-23.5,-48.5,-27,-51,-31,-54,-34.5,-57.5,-36.5,-58,-38.5,-62,-39,-65,-41,-65,-45,-67.5,-46.5,-66,-48,-69,-51.5,-68.5,-54.5,-71,-54,-74,-51.5,-75,-47,-73,-42,-73.5,-37,-71.5,-31,-70.5,-24,-70,-18.5,-75,-15,-77,-12,-79.5,-7,-81,-5,-80,-2.5,-80,0.5,-78.5,2.5,-77.5,6],
    // Africa
    [-17,21,-16,24,-13,27.6,-9.6,30,-9.6,32.5,-6,35.6,-2,35.3,3,36.9,10,37.2,11,33.6,15,32.2,19,30.4,20,32.5,25,31.7,29,31,31,31.5,32.3,31.2,32.6,29.9,33.5,27.8,35.5,24,37,21,38.5,18,40,15.5,43,12.5,43.5,11.4,47,11.2,51,12,51,10.5,48,5,45.5,2,42,-1.5,40,-4,39,-7,40,-11,40.5,-15,37,-18,35,-20,35.5,-24,33,-25.5,32.5,-28.5,30,-31.5,26.5,-34,22,-34.4,18.5,-34.4,18,-32,16,-28.5,14.5,-23,12,-18,13.5,-12,12.5,-6,9.5,-2,9.5,3.5,8.5,4.5,6,4.3,4,6.3,1,6,-2,5,-4.5,5.2,-7.5,4.4,-9.5,6,-12.5,7.5,-13.5,9.5,-15,11,-17,14.7,-16.5,16,-16.2,19],
    // Europe and Asia
    [-9,37,-9.4,39,-8.8,43.2,-2,43.4,-1.3,46,-4.6,48.4,-1.6,48.7,1.6,50.9,4,52,7,53.6,8.6,54,8.2,56.8,10.5,57.6,10.2,55,11,54,14,54,18.5,54.6,21,55.2,21,57.2,24.2,57.6,23.5,59.2,28,59.6,30,60,28.5,60.6,24,60,21.5,60.6,21.3,62.8,25,65,24,65.8,22,65.6,21.2,64.2,17.5,62,17.2,60.6,18.8,59.8,16.6,57,15.8,56.2,14.2,55.5,12.9,55.5,12.4,56.6,11.4,58,10.8,59.2,9.4,58.9,7,58,5.5,59,5,61.6,8,63.6,12,65.6,14.6,68,18.6,69.8,24,71,28,71,31,70,33.5,69.3,40.5,67,44,66.2,44,68.4,53,68.6,60,69,68,68.6,68,71.5,72.5,72.8,80,73,87,75.4,95,76,104,77.6,113,75.6,113,73.8,127,73,131,71,140,72.5,150,71.6,160,69.8,170,69.6,178,69,180,68.6,180,65,177,62.5,170,60,164,59.6,163,56,162,54.6,160,52,156.6,51,156,57,160,61.5,155,59.5,150,59.5,143,59.2,138,56.2,137.5,54,141,53,140.5,50,138,46.6,135,43.5,131,42.8,129.5,41.5,127.6,39.8,129.3,37,129,35.2,126.5,34.5,126.5,37,125,38,124.5,40,121.5,40.6,121,39,118,39,119,37.5,122.5,37,120,35,121,32,122,30,120,27,117,23.5,113.5,22.4,110,21,108,21.5,106.5,19,108.8,15.5,109.2,12,106.5,10,105,8.8,104.8,10.2,102.5,12,100.5,13.2,99.5,10,101,6.5,103.5,2.5,104,1.3,101.5,2.6,100.3,5.5,98.3,8,98.5,12,97.5,16.5,94.5,16.5,94,19.5,92,22,90,22,87,21,85,19.5,80.5,15.5,80,13,79.5,10.3,77.5,8,76,10.5,74.5,14.5,73,19,72.5,21,70,21,69,22.3,67,24.5,62,25.2,57.5,25.5,56.5,27,54,26.6,51,28,49,30,48,29.5,50,26.5,51.5,24.3,54.5,24.2,56.2,25.8,56.6,24.5,59.8,22.4,57.8,19,55,17,52,15.5,48,14,45,12.8,43.3,12.8,42.5,16,40,20,38.5,23.5,36,26.5,35,28,34.3,27.8,32.6,29.9,32.3,31.2,34.5,31.5,35.5,34,36,36.5,34,36.2,31,36.6,28.5,36.7,27,37.5,26.5,39.5,29,41,31.5,41.2,35,42,38,41,41.5,41.5,41,43,38.5,44.5,37,45.3,35,45,33.5,46,31,46.5,29.5,45.3,28.5,43.5,28,42,28.6,41.2,26,40.8,24,40.8,23,40,24,38,23,36.5,21.5,37,20,39.5,19.5,41.8,16,43.5,13.8,45.5,12.3,45.3,12.5,44,14,42.5,16,41.8,18.5,40.2,17,40.3,16.5,39,16,38,15.7,40,14,40.8,12.3,41.8,10.5,43,8.8,44.4,7.5,43.8,3.5,43.3,3,42,0.5,40.5,-0.3,38.8,-2,36.8,-5.3,36.2,-6.5,37],
    [-180,68.6,-170,66,-173,64.5,-180,65],                                                            // eastern tip of Asia
    [-44,60,-51,64,-54,69,-58,75,-68,77.5,-60,81.5,-40,83,-22,81.5,-19,76,-22,71,-28,68,-38,65.5],     // Greenland
    [-65,62,-62,66.5,-68,70,-78,73.5,-88,73.5,-88,71,-80,68,-74,64.5,-70,62.5],                        // Baffin Island
    [-117,70.5,-112,73,-104,73.2,-101,71,-106,69.2,-114,69.3],                                         // Victoria Island
    [-90,76.5,-80,76.3,-75,78.5,-62,82,-70,83,-90,81.5,-95,79],                                        // Ellesmere Island
    [-24,65.5,-22,66.4,-16,66.5,-13.5,65,-18,63.4,-22.5,63.8],                                         // Iceland
    [-5.5,50,1.3,51.2,1.7,52.8,0,53.6,-1.6,55.5,-2,57.6,-3.5,58.6,-5.5,58.5,-6,56.5,-4.8,54.8,-3,53.5,-4.5,53,-5.2,51.8,-3,51.4], // Great Britain
    [-6,52.2,-6,54,-7.5,55.2,-10,54,-10,52,-8.5,51.5],                                                 // Ireland
    [12.5,38,15.6,38.2,15.2,36.7,12.5,37.6],                                                           // Sicily
    [8.2,41.2,9.7,41,9.6,39.2,8.5,39],                                                                 // Sardinia
    [51.5,71.5,56,73.5,60,76,68,77,62,75,57,72,55.5,70.7],                                             // Novaya Zemlya
    [11,79,16,80,24,80.3,27,79.5,20,78,17,76.7,14,77.5],                                               // Svalbard
    [-85,22,-80,23.2,-75.5,20.7,-74.2,20.2,-77.5,19.9,-78,21.5,-82,22.3],                              // Cuba
    [-74.4,19.8,-70,19.8,-68.4,18.6,-71.5,17.8,-74.4,18.4],                                            // Hispaniola
    [49.2,-12,50.4,-15.5,47.8,-24.8,45.2,-25.5,43.5,-22,44.3,-16.5,47,-14.5],                          // Madagascar
    [80,9.7,81.8,7.5,81,6.1,80,6.2,79.8,8],                                                            // Sri Lanka
    [130,31.5,131.5,31.3,132,33.8,135.5,33.6,137,34.6,140,35,141,38,141.8,40.5,141.3,41.4,140,40.6,139.5,38.2,137,36.9,136,35.8,133,35.5,131,34.4,130,33.3], // Japan
    [140.2,41.6,141.2,43,141.7,45.3,145.2,44,145,43,143,42,141,42.4],                                  // Hokkaido
    [142,46,143.5,49,143,53.5,142,53,142.2,49,141.8,46.5],                                             // Sakhalin
    [120.3,22,121.8,25,122,24,121,22],                                                                 // Taiwan
    [120.5,18.5,122.2,18.3,122,14,124,13,121,13.7,120,16],                                             // Luzon
    [122,7,125.5,9.7,126.5,7,125.5,5.8,123.5,7.5],                                                     // Mindanao
    [95.3,5.6,98,4,103.5,-1,106,-3.5,105.5,-5.8,102,-4.5,98.7,0.5],                                    // Sumatra
    [105.5,-6.5,108.5,-6.6,111,-6.5,114.5,-7.5,114.3,-8.6,108,-7.8,105.5,-6.9],                        // Java
    [109,1.5,111.5,2.5,113.5,4,117,7,119,5.2,117.8,1.8,116.5,-2.5,116.2,-4,114.5,-3.8,111.5,-3.2,110,-1.5,109,0.2], // Borneo
    [119.5,-5.5,120.5,-3,120,0.5,123,1,125,1.5,121.5,-1,123,-4.5,121.5,-4.8],                          // Sulawesi
    [131,-0.8,134,-0.8,135.5,-3.3,138,-1.7,141,-2.6,145,-4.3,147.5,-6,146.5,-8.5,150.5,-10.5,147,-9.8,143,-9,141,-9.2,138.5,-8.2,138,-5.5,135,-4.4,133,-4,132,-2.8], // New Guinea
    [114,-22,113.5,-26,115,-31.5,115.2,-34.3,118,-35,123.5,-34,129,-31.7,131.5,-31.5,134.5,-33,136,-35,138,-33.5,138,-35.6,140,-38,143.5,-38.8,146.5,-39,150,-37.5,151.3,-33.5,153.2,-30,153.2,-25.5,150.8,-23.3,149,-21,146.3,-19,145.5,-15,143.5,-14.2,142.5,-10.8,141.7,-13,141,-17.3,139.5,-17.5,136,-15.5,136.8,-12.2,133,-11.5,130.8,-12.5,129.5,-15,127,-14,124.5,-16.3,122.2,-18,119,-20,116.5,-20.7], // Australia
    [144.7,-40.8,148.2,-40.9,148,-43.2,146,-43.6,145.2,-42.2],                                         // Tasmania
    [172.7,-34.5,175.5,-36.5,178.3,-37.7,177,-39.5,175.3,-41.6,174.7,-39.8,173.8,-39.3,174.7,-37.5],   // New Zealand, North Island
    [172.7,-40.6,174.2,-41.7,172.8,-43.8,171,-45.5,169,-46.6,166.5,-45.8,168.5,-44,171,-42.3],         // New Zealand, South Island
    [-180,-90,-180,-78,-160,-78,-150,-76.5,-130,-74.5,-110,-74.5,-100,-72.5,-80,-73,-70,-71,-62,-64.5,-57,-63.5,-60,-67,-62,-72,-60,-75.5,-40,-78,-25,-75,-12,-71.5,0,-70,15,-69.5,30,-69,40,-68,52,-66,60,-67.5,70,-68.5,78,-67,90,-66.5,100,-66,110,-66.5,120,-67,135,-66,145,-67.5,160,-70,170,-72,165,-77.5,180,-78,180,-90] // Antarctica
  ];
  var WATER = [
    [47,45,51,46.8,53,45,51,44.5,50.5,42,53,40.5,54,37.5,51,36.8,49,37.5,49.5,40.5,47.5,43]            // Caspian Sea
  ];

  var $ = function (id) { return document.getElementById(id); };
  var THREE, renderer, scene, camera, viewEl, labelsEl;
  var world, formula, formulaText = "";
  var opts = { auto: true, earliestFree: false, orbits: true };
  var sats3d = [], satPos = [], links = null, apIndex = 0, gs;
  var packets = [], effects = [], labels = [];
  var linkLines, routeLines, beam, orbitRings = [];
  var packetGeo, cubeGeo, glowTex, ringTex;
  var orbitTime = 0, animTime = 0, linkTimer = 0, nextRoundAt = 0, runRequested = false, chipTimer = 0;
  var cam = { theta: 0.15, phi: 1.05, radius: 5.4, zoomed: false };
  var visible = true, lastDecision = null, logLines = [], worldToken = 0, lastFrame = 0;

  function fmt(x) {
    if (x !== x) return "NaN";
    if (x === Infinity) return "+∞";
    if (x === -Infinity) return "−∞";
    return (Math.abs(x) >= 10000 || (Math.abs(x) < 0.01 && x !== 0)) ? x.toExponential(1) : (+x.toFixed(2)).toString();
  }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  // ── Textures ───────────────────────────────────────────────────────────
  function makeEarthTexture() {
    var W = 2048, H = 1024, c = document.createElement("canvas"), g = c.getContext("2d");
    c.width = W; c.height = H;
    function X(lon) { return (lon + 180) / 360 * W; }
    function Y(lat) { return (90 - lat) / 180 * H; }
    function trace(p) {
      g.beginPath();
      for (var k = 0; k < p.length; k += 2) { if (k) g.lineTo(X(p[k]), Y(p[k + 1])); else g.moveTo(X(p[k]), Y(p[k + 1])); }
      g.closePath();
    }
    var ocean = g.createLinearGradient(0, 0, 0, H);
    ocean.addColorStop(0, "#16406b"); ocean.addColorStop(0.5, "#0f4f86"); ocean.addColorStop(1, "#16406b");
    g.fillStyle = ocean; g.fillRect(0, 0, W, H);
    // Land colour by latitude: ice, forest, desert belt, tropics, and back.
    var land = g.createLinearGradient(0, 0, 0, H);
    [[0, "#f2f6f8"], [0.08, "#e6ecee"], [0.14, "#8fa58c"], [0.22, "#5b8f5e"], [0.31, "#9a9a62"], [0.37, "#c2a56b"],
     [0.43, "#7d9a57"], [0.5, "#3f8450"], [0.56, "#5f9455"], [0.65, "#b99a63"], [0.72, "#6f9659"], [0.8, "#8fa58c"],
     [0.86, "#e6ecee"], [1, "#f7fafb"]].forEach(function (s) { land.addColorStop(s[0], s[1]); });
    g.lineJoin = "round";
    LAND.forEach(function (p) { trace(p); g.fillStyle = land; g.fill(); g.lineWidth = 3; g.strokeStyle = land; g.stroke(); });
    LAND.forEach(function (p) { trace(p); g.lineWidth = 1; g.strokeStyle = "rgba(255,255,255,.22)"; g.stroke(); });
    WATER.forEach(function (p) { trace(p); g.fillStyle = ocean; g.fill(); });
    g.fillStyle = "rgba(236,243,248,.5)"; g.fillRect(0, 0, W, Y(83));          // Arctic sea ice
    g.strokeStyle = "rgba(255,255,255,.07)"; g.lineWidth = 1.5;                 // graticule every 30 degrees
    for (var lon = -180; lon <= 180; lon += 30) { g.beginPath(); g.moveTo(X(lon), 0); g.lineTo(X(lon), H); g.stroke(); }
    for (var lat = -60; lat <= 60; lat += 30) { g.beginPath(); g.moveTo(0, Y(lat)); g.lineTo(W, Y(lat)); g.stroke(); }
    var tex = new THREE.CanvasTexture(c);
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }
  function radialTexture(ring) {
    var c = document.createElement("canvas"), g = c.getContext("2d"); c.width = c.height = 128;
    var gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    if (ring) { gr.addColorStop(0, "rgba(255,255,255,0)"); gr.addColorStop(0.72, "rgba(255,255,255,0)"); gr.addColorStop(0.86, "rgba(255,255,255,1)"); gr.addColorStop(1, "rgba(255,255,255,0)"); }
    else { gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.25, "rgba(255,255,255,.55)"); gr.addColorStop(1, "rgba(255,255,255,0)"); }
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  // ── Building the scene ─────────────────────────────────────────────────
  function buildScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(42, 1, 0.1, 300);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    var sun = new THREE.DirectionalLight(0xffffff, 0.95); sun.position.set(5, 3, 4); scene.add(sun);

    // Stars
    var sp = new Float32Array(900 * 3), rnd = mulberry32(7);
    for (var i = 0; i < 900; i++) {
      var u = rnd() * 2 - 1, a = rnd() * TWO_PI, r = Math.sqrt(1 - u * u);
      sp[i * 3] = 120 * r * Math.cos(a); sp[i * 3 + 1] = 120 * u; sp[i * 3 + 2] = 120 * r * Math.sin(a);
    }
    var sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
    scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: 1.3, sizeAttenuation: false, transparent: true, opacity: 0.7 })));

    // Earth and a thin atmosphere shell
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 72, 48),
      new THREE.MeshPhongMaterial({ map: makeEarthTexture(), shininess: 10, specular: 0x1a2a3a })));
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.045, 48, 32),
      new THREE.MeshBasicMaterial({ color: 0x5aa9ff, transparent: true, opacity: 0.1, side: THREE.BackSide, depthWrite: false })));

    glowTex = radialTexture(false); ringTex = radialTexture(true);
    packetGeo = new THREE.SphereGeometry(0.026, 12, 10);
    cubeGeo = new THREE.BoxGeometry(0.03, 0.03, 0.03);

    // Ground user
    var g0 = latLon(GS_LAT, GS_LON, 1.004);
    gs = new THREE.Vector3(g0.x, g0.y, g0.z);
    var pin = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 10), new THREE.MeshBasicMaterial({ color: 0xffd166 }));
    pin.position.copy(gs); scene.add(pin);
    var halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: 0xffd166, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    halo.scale.set(0.16, 0.16, 1); halo.position.copy(gs); scene.add(halo);
    addLabel(el("div", "v3-label gs", "Ground user"), gs, 8);

    // Orbit rings (one per orbital plane)
    for (var p = 0; p < N_PLANES; p++) {
      var pts = [];
      for (var k = 0; k < 128; k++) {
        var ang = k / 128 * TWO_PI;
        pts.push(new THREE.Vector3(ORBIT_R * Math.cos(ang), ORBIT_R * Math.sin(ang) * Math.sin(INCLINATION), ORBIT_R * Math.sin(ang) * Math.cos(INCLINATION)));
      }
      var ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x9fb4d8, transparent: true, opacity: 0.16 }));
      orbitRings.push(ring); scene.add(ring);
    }

    // Satellites
    var bodyGeo = new THREE.BoxGeometry(0.045, 0.045, 0.06), panelGeo = new THREE.BoxGeometry(0.09, 0.004, 0.04);
    var dishGeo = new THREE.SphereGeometry(0.014, 10, 8);
    for (var s = 0; s < N_SAT; s++) {
      var grp = new THREE.Group();
      var bodyMat = new THREE.MeshPhongMaterial({ color: 0xd5dbe6, emissive: 0x1f7a4d, emissiveIntensity: 0.6 });
      grp.add(new THREE.Mesh(bodyGeo, bodyMat));
      var panelMat = new THREE.MeshPhongMaterial({ color: 0x2f66d6, emissive: 0x102a66, shininess: 60 });
      var left = new THREE.Mesh(panelGeo, panelMat), right = new THREE.Mesh(panelGeo, panelMat);
      left.position.x = -0.07; right.position.x = 0.07; grp.add(left, right);
      var dish = new THREE.Mesh(dishGeo, new THREE.MeshBasicMaterial({ color: 0xf2f4f8 })); dish.position.z = 0.036; grp.add(dish);
      var queue = new THREE.Group(); grp.add(queue);
      scene.add(grp);
      var lab = el("div", "v3-label sat");
      lab.appendChild(el("span", "n", "S" + s));
      var bar = el("span", "bar"), fill = el("i"); bar.appendChild(fill); lab.appendChild(bar);
      var chip = el("span", "chip"); lab.appendChild(chip);
      addLabel(lab, grp.position, 12);
      sats3d.push({ group: grp, bodyMat: bodyMat, queue: queue, label: lab, fill: fill, chip: chip, bump: 0 });
      satPos.push(grp.position);
    }

    // Inter-satellite links, the planned route of moving tasks, and the uplink beam
    var lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N_SAT * 3 * 6), 3));
    linkLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x74d4ff, transparent: true, opacity: 0.24, depthWrite: false }));
    linkLines.frustumCulled = false; scene.add(linkLines);
    var rg = new THREE.BufferGeometry();
    rg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(96 * 6), 3));
    rg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(96 * 6), 3));
    routeLines = new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
    routeLines.frustumCulled = false; scene.add(routeLines);
    var bg = new THREE.BufferGeometry(); bg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    beam = new THREE.Line(bg, new THREE.LineDashedMaterial({ color: 0xffd166, dashSize: 0.045, gapSize: 0.035, transparent: true, opacity: 0.65 }));
    beam.frustumCulled = false; scene.add(beam);
  }

  function addLabel(node, posRef, lift) { labelsEl.appendChild(node); var l = { el: node, pos: posRef, lift: lift, shown: true }; labels.push(l); return l; }
  function removeLabel(l) { var i = labels.indexOf(l); if (i >= 0) labels.splice(i, 1); if (l.el.parentNode) l.el.parentNode.removeChild(l.el); }

  // ── Per-frame updates of the constellation ─────────────────────────────
  function updateSats(dt) {
    for (var p = 0; p < N_PLANES; p++) orbitRings[p].rotation.y = planeAngle(p, orbitTime);
    for (var i = 0; i < N_SAT; i++) {
      var s = sats3d[i];
      satPosition(i, orbitTime, s.group.position);
      s.group.lookAt(0, 0, 0);                       // local +z points at the Earth
      if (s.bump > 0) s.bump = Math.max(0, s.bump - dt * 2.5);
      s.group.scale.setScalar(1 + 0.6 * Math.sin(s.bump * Math.PI));
    }
  }
  function refreshLinks() {
    links = buildLinks(satPos);
    var ap = pickAccessPoint(satPos, gs);
    if (ap !== apIndex) { sats3d[apIndex].label.classList.remove("ap"); apIndex = ap; }
    sats3d[apIndex].label.classList.add("ap");
  }
  function updateLines() {
    var a = linkLines.geometry.attributes.position.array, n = 0, i;
    for (i = 0; i < links.pairs.length && n < N_SAT * 3; i++, n++) {
      var p = satPos[links.pairs[i][0]], q = satPos[links.pairs[i][1]];
      a[n * 6] = p.x; a[n * 6 + 1] = p.y; a[n * 6 + 2] = p.z; a[n * 6 + 3] = q.x; a[n * 6 + 4] = q.y; a[n * 6 + 5] = q.z;
    }
    linkLines.geometry.setDrawRange(0, n * 2); linkLines.geometry.attributes.position.needsUpdate = true;

    var b = beam.geometry.attributes.position.array, t = satPos[apIndex];
    b[0] = gs.x; b[1] = gs.y; b[2] = gs.z; b[3] = t.x; b[4] = t.y; b[5] = t.z;
    beam.geometry.attributes.position.needsUpdate = true; beam.computeLineDistances();

    // The part of each route that a moving task still has to fly.
    var rp = routeLines.geometry.attributes.position.array, rc = routeLines.geometry.attributes.color.array, m = 0;
    for (i = 0; i < packets.length; i++) {
      var job = packets[i], v = job.vis;
      if (job.state !== "routing" || !v.path) continue;
      var from = v.group.position;
      for (var k = Math.max(1, v.seg); k < v.path.length && m < 96; k++, m++) {
        var to = satPos[v.path[k]];
        rp[m * 6] = from.x; rp[m * 6 + 1] = from.y; rp[m * 6 + 2] = from.z; rp[m * 6 + 3] = to.x; rp[m * 6 + 4] = to.y; rp[m * 6 + 5] = to.z;
        for (var c = 0; c < 2; c++) { rc[m * 6 + c * 3] = v.color.r; rc[m * 6 + c * 3 + 1] = v.color.g; rc[m * 6 + c * 3 + 2] = v.color.b; }
        from = to;
      }
    }
    routeLines.geometry.setDrawRange(0, m * 2);
    routeLines.geometry.attributes.position.needsUpdate = true; routeLines.geometry.attributes.color.needsUpdate = true;
  }

  // ── Tasks on screen ────────────────────────────────────────────────────
  var TRAIL = 18;
  function spawnPacket(job) {
    var color = new THREE.Color().setHSL(((job.id * 47 + 190) % 360) / 360, 0.85, 0.64);
    var group = new THREE.Group();
    var core = new THREE.Mesh(packetGeo, new THREE.MeshBasicMaterial({ color: color }));
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: color, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    glow.scale.set(0.2, 0.2, 1);
    group.add(core, glow); group.position.copy(gs); scene.add(group);
    var tg = new THREE.BufferGeometry(), tp = new Float32Array(TRAIL * 3), tc = new Float32Array(TRAIL * 3);
    for (var i = 0; i < TRAIL; i++) {
      var f = 1 - i / TRAIL; tp[i * 3] = gs.x; tp[i * 3 + 1] = gs.y; tp[i * 3 + 2] = gs.z;
      tc[i * 3] = color.r * f; tc[i * 3 + 1] = color.g * f; tc[i * 3 + 2] = color.b * f;
    }
    tg.setAttribute("position", new THREE.BufferAttribute(tp, 3)); tg.setAttribute("color", new THREE.BufferAttribute(tc, 3));
    var trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ vertexColors: true, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }));
    trail.frustumCulled = false; scene.add(trail);
    job.vis = { group: group, core: core, glow: glow, trail: trail, color: color, tmp: new THREE.Vector3(),
                segs: [], seg: 0, t: 0, onDone: null, path: null,
                label: addLabel(el("div", "v3-label pk", job.name), group.position, 6) };
    packets.push(job);
  }
  function removePacket(job) {
    var v = job.vis; if (!v) return;
    scene.remove(v.group); scene.remove(v.trail);
    v.core.material.dispose(); v.glow.material.dispose(); v.trail.geometry.dispose(); v.trail.material.dispose();
    removeLabel(v.label);
    packets.splice(packets.indexOf(job), 1); job.vis = null;
  }
  function recolor(job, hex) {
    var v = job.vis; v.color.set(hex); v.core.material.color.set(hex); v.glow.material.color.set(hex);
    var tc = v.trail.geometry.attributes.color.array;
    for (var i = 0; i < TRAIL; i++) { var f = 1 - i / TRAIL; tc[i * 3] = v.color.r * f; tc[i * 3 + 1] = v.color.g * f; tc[i * 3 + 2] = v.color.b * f; }
    v.trail.geometry.attributes.color.needsUpdate = true;
  }
  // Where a task hovers while it waits at its access point: just below the satellite.
  function waitPos(job) {
    return job.vis.tmp.copy(satPos[job.ap]).multiplyScalar(1 - 0.05 * (1 + Math.min(job.slot, 6)));
  }
  function setPath(job, segs, onDone) { var v = job.vis; v.segs = segs; v.seg = 0; v.t = 0; v.onDone = onDone || null; }

  function updatePackets(dt) {
    var count = {}, n, job;
    for (n = 0; n < packets.length; n++) {
      job = packets[n];
      if (job.state === "waiting" || job.state === "uplink") { job.slot = count[job.ap] || 0; count[job.ap] = job.slot + 1; }
    }
    var list = packets.slice();                      // callbacks may remove packets
    for (n = 0; n < list.length; n++) {
      job = list[n];
      var v = job.vis, p = v.group.position;
      if (v.segs.length) {
        var sg = v.segs[v.seg];
        v.t += dt / sg.dur;
        var t = v.t >= 1 ? 1 : v.t;
        p.lerpVectors(sg.a(), sg.b(), t * t * (3 - 2 * t));
        if (v.t >= 1) {
          v.seg++; v.t = 0;
          if (v.seg >= v.segs.length) { v.segs = []; var cb = v.onDone; v.onDone = null; if (cb) cb(); }
        }
      } else if (job.state === "waiting") {
        p.copy(waitPos(job));
      }
      if (!job.vis) continue;                        // removed by its callback
      var tp = v.trail.geometry.attributes.position.array;
      tp.copyWithin(3, 0, (TRAIL - 1) * 3); tp[0] = p.x; tp[1] = p.y; tp[2] = p.z;
      v.trail.geometry.attributes.position.needsUpdate = true;
    }
  }
  function updateEffects(dt) {
    for (var i = effects.length - 1; i >= 0; i--) {
      var e = effects[i]; e.t += dt / e.dur;
      if (e.t >= 1) { e.end(); effects.splice(i, 1); } else e.update(e.t);
    }
  }
  function pulse(posRef, hex, size) {
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, color: hex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    scene.add(sp);
    effects.push({ t: 0, dur: 0.9,
      update: function (k) { sp.position.copy(posRef); var s = size * (0.3 + k); sp.scale.set(s, s, 1); sp.material.opacity = 1 - k; },
      end: function () { scene.remove(sp); sp.material.dispose(); } });
  }

  // ── The life of one task ───────────────────────────────────────────────
  function log(text, cls) {
    logLines.push({ text: text, cls: cls || "" });
    if (logLines.length > 7) logLines.shift();
    var box = $("v3-log"); box.textContent = "";
    logLines.forEach(function (l) { box.appendChild(el("div", l.cls, l.text)); });
  }

  function createTask() {
    var job = newJob(world);
    job.ap = apIndex;
    world.jobs.push(job); world.created++;
    spawnPacket(job);
    var gsFn = function () { return gs; };
    setPath(job, [{ a: gsFn, b: function () { return waitPos(job); }, dur: 0.9 }], function () {
      job.state = "waiting";
      nextRoundAt = Math.max(nextRoundAt, animTime + 0.35);
      log(job.name + " reached access point S" + job.ap + " (arrival " + job.arrivalTime + ", needs " + duration(job) + " min)");
      renderStats();
    });
    renderStats();
  }

  function waitingJobs() { return world.jobs.filter(function (j) { return j.state === "waiting"; }); }

  // One decoder round on the tasks that are waiting now.  Returns false when
  // there was nothing to do or nothing could be placed.
  function decodeRound() {
    var waiting = waitingJobs();
    if (!waiting.length) return false;
    advanceClock(world.sats, world.tau);
    var best = pickBest(waiting, world.sats, formula, opts.earliestFree);
    if (!best) { rejectAll(waiting); return false; }

    var rows = world.sats.map(function (s) { return { reason: infeasibleReason(best.job, s), score: formula(best.job, s) }; });
    commit(best.job, best.sat);
    world.scheduled++; world.responseSum += best.job.end - best.job.arrivalTime;
    lastDecision = { job: best.job, rows: rows, winner: best.sat.index, rejected: false };
    dispatch(best.job);
    showChips(); renderTable(); renderStats(); refreshSatLooks();
    return true;
  }

  function dispatch(job) {
    var v = job.vis, target = job.sat.index;
    var path = route(links, job.ap, target), hold = v.group.position.clone(), segs = [];
    job.state = "routing"; v.path = path;
    var holdFn = function () { return hold; };
    segs.push({ a: holdFn, b: holdFn, dur: 0.5 });                         // pause: read the scores first
    function node(i) { return function () { return satPos[path[i]]; }; }
    if (path.length === 1) segs.push({ a: holdFn, b: node(0), dur: 0.35 });
    else for (var i = 1; i < path.length; i++) segs.push({ a: i === 1 ? holdFn : node(i - 1), b: node(i), dur: 0.55 });
    pulse(satPos[target], 0xffd166, 0.5);
    var hops = path.length - 1;
    log(job.name + " → " + job.sat.name + " (score " + fmt(lastDecision.rows[target].score) + ", " +
        (hops === 0 ? "runs on its access point" : hops + (hops === 1 ? " hop" : " hops") + ": " + path.map(function (i) { return "S" + i; }).join(" → ")) +
        "), minutes " + fmt(job.start) + "–" + fmt(job.end), "ok");
    setPath(job, segs, function () {
      job.state = "queued";
      var s3 = sats3d[target], n = s3.queue.children.length;
      if (n < 10) {
        var cube = new THREE.Mesh(cubeGeo, new THREE.MeshBasicMaterial({ color: v.color.clone() }));
        cube.position.z = -(0.07 + n * 0.038);                             // stacked on the side away from the Earth
        s3.queue.add(cube);
      }
      s3.bump = 1;
      removePacket(job);
    });
  }

  function rejectAll(waiting) {
    var first = waiting[0];
    var rows = world.sats.map(function (s) { return { reason: infeasibleReason(first, s), score: formula(first, s) }; });
    lastDecision = { job: first, rows: rows, winner: -1, rejected: true };
    waiting.forEach(function (job) {
      var e = 0, w = 0;
      world.sats.forEach(function (s) { var r = infeasibleReason(job, s); if (r === "energy") e++; else if (r === "sunset") w++; });
      var why = (e + w === world.sats.length)
        ? "no satellite is feasible (energy on " + e + ", sunset on " + w + ")"
        : "its score is −∞ or NaN on every feasible satellite";
      log(job.name + " rejected: " + why, "bad");
      job.state = "rejected"; world.rejected++;
      recolor(job, 0xff5a4d);
      var hold = job.vis.group.position.clone();
      setPath(job, [{ a: function () { return hold; }, b: function () { return hold; }, dur: 0.5 },
                    { a: function () { return hold; }, b: function () { return gs; }, dur: 1.0 }],
              function () { pulse(gs, 0xff5a4d, 0.35); removePacket(job); });
    });
    showChips(); renderTable(); renderStats();
  }

  function stepScheduler() {
    if (!(opts.auto || runRequested) || animTime < nextRoundAt) return;
    if (!waitingJobs().length) {
      if (!world.jobs.some(function (j) { return j.state === "uplink"; })) runRequested = false;
      return;
    }
    decodeRound();
    nextRoundAt = animTime + (opts.auto ? 0.75 : 1.1);
  }

  // ── Panels around the picture ──────────────────────────────────────────
  function satUsable(s) {   // could it still take the shortest possible job (6 minutes)?
    return s.remainingEnergy >= 6 * s.rate && Math.max(s.freeAt, world.tau) + 6 <= s.operationalUntil;
  }
  function refreshSatLooks() {
    for (var i = 0; i < N_SAT; i++) {
      var s = world.sats[i], v = sats3d[i], share = Math.max(0, s.remainingEnergy / s.energy0);
      var usable = satUsable(s), low = usable && share < 0.35;
      v.fill.style.width = Math.round(share * 100) + "%";
      v.label.classList.toggle("low", low); v.label.classList.toggle("out", !usable);
      v.bodyMat.emissive.set(!usable ? 0x444a55 : (low ? 0x9a5800 : 0x1f7a4d));
    }
  }
  function showChips() {
    var d = lastDecision;
    for (var i = 0; i < N_SAT; i++) {
      var chip = sats3d[i].chip, r = d.rows[i];
      chip.textContent = r.reason ? "✕ " + r.reason : fmt(r.score);
      chip.className = "chip" + (r.reason ? " bad" : "") + (i === d.winner ? " win" : "");
    }
    chipTimer = opts.auto ? 2.6 : 6;
  }
  function clearChips() { for (var i = 0; i < N_SAT; i++) { sats3d[i].chip.textContent = ""; sats3d[i].chip.className = "chip"; } }

  function renderStats() {
    var w = world, pending = w.created - w.scheduled - w.rejected;
    var energy = 0; w.sats.forEach(function (s) { energy += s.spent; });
    $("v3-stats").innerHTML =
      "<div><span>World minute τ</span><b>" + w.tau + "</b></div>" +
      "<div><span>Tasks created</span><b>" + w.created + "</b></div>" +
      "<div><span>Scheduled / rejected / waiting</span><b>" + w.scheduled + " / " + w.rejected + " / " + pending + "</b></div>" +
      "<div><span>Completion ratio C</span><b>" + (w.scheduled + w.rejected ? (w.scheduled / (w.scheduled + w.rejected)).toFixed(2) : "–") + "</b></div>" +
      "<div><span>Mean response R</span><b>" + (w.scheduled ? (w.responseSum / w.scheduled).toFixed(1) + " min" : "–") + "</b></div>" +
      "<div><span>Energy spent E</span><b>" + fmt(energy) + "</b></div>";
    var noAuto = !opts.auto;
    $("v3-step").disabled = !noAuto; $("v3-run").disabled = !noAuto;
  }
  function renderTable() {
    var d = lastDecision, t = $("v3-sats");
    var h = "<tr><th>Satellite</th><th class='num'>Energy left</th><th class='num'>Free at</th><th class='num'>Sunset</th><th class='num'>Tasks</th>" +
            "<th class='num'>" + (d ? "Score for " + d.job.name : "Score") + "</th><th>" + (d ? "Could it take " + d.job.name + "?" : "Feasible?") + "</th></tr>";
    world.sats.forEach(function (s, i) {
      var r = d ? d.rows[i] : null, win = d && d.winner === i;
      h += "<tr" + (win ? " class='hl'" : "") + "><td>" + s.name + "</td>" +
           "<td class='num'>" + fmt(s.remainingEnergy) + " / " + s.energy0 + "</td>" +
           "<td class='num'>" + fmt(Math.max(s.freeAt, world.tau)) + "</td><td class='num'>" + s.operationalUntil + "</td>" +
           "<td class='num'>" + s.completedTasks + "</td>" +
           "<td class='num" + (r && r.reason ? " infeasible" : "") + (win ? " best" : "") + "'>" + (r ? fmt(r.score) : "–") + "</td>" +
           "<td>" + (!r ? "–" : win ? "<span class='pill ok'>chosen</span>" : r.reason === "energy" ? "<span class='pill bad'>no: energy</span>" :
                     r.reason === "sunset" ? "<span class='pill bad'>no: sunset</span>" : "yes") + "</td></tr>";
    });
    t.innerHTML = h;
    var p = $("v3-decision");
    if (!d) p.textContent = "No decision yet. Press Create task.";
    else if (d.rejected) p.textContent = "Last decision: " + d.job.name + " (needs " + duration(d.job) + " minutes) could not be placed. The decoder stops, and every task still waiting is rejected.";
    else p.textContent = "Last decision: " + d.job.name + " (arrival " + d.job.arrivalTime + ", execution " + d.job.executionTime + " + transfer " + d.job.transferTime +
                         " = " + duration(d.job) + " minutes) went to " + d.job.sat.name + ", minutes " + fmt(d.job.start) + " to " + fmt(d.job.end) +
                         ". The scores below were read just before the commit; the other columns show the state after it.";
  }

  function resetWorld(seed) {
    worldToken++;
    while (packets.length) removePacket(packets[0]);
    sats3d.forEach(function (s) { while (s.queue.children.length) { var c = s.queue.children[0]; s.queue.remove(c); c.material.dispose(); } });
    world = newWorld(seed, $("v3-world").value);
    lastDecision = null; runRequested = false; logLines = [];
    clearChips(); refreshSatLooks(); renderStats(); renderTable();
    log("World " + seed + ": " + N_SAT + " satellites, all idle, no task yet.");
  }

  function setFormula(text) {
    var st = $("v3-status");
    try { formula = window.OnboardingFormula.parse(text); formulaText = text; st.textContent = ""; st.className = "status"; return true; }
    catch (e) { st.textContent = e.message + " (still using: " + formulaText + ")"; st.className = "status err"; return false; }
  }

  // ── Camera and labels ──────────────────────────────────────────────────
  function fitRadius() {
    var half = Math.tan(camera.fov * Math.PI / 360);
    return Math.max(5.2, 2.0 / (half * Math.min(1, camera.aspect)));
  }
  function updateCamera() {
    var r = cam.radius, sp = Math.sin(cam.phi);
    camera.position.set(r * sp * Math.cos(cam.theta), r * Math.cos(cam.phi), r * sp * Math.sin(cam.theta));
    camera.lookAt(0, 0, 0);
  }
  function zoom(f) { cam.radius = Math.min(14, Math.max(2.4, cam.radius * f)); cam.zoomed = true; }
  function resetView() { cam.theta = 0.15; cam.phi = 1.05; cam.zoomed = false; cam.radius = fitRadius(); }
  var tmpV = null;
  function hiddenByEarth(p) {
    var o = camera.position, dx = p.x - o.x, dy = p.y - o.y, dz = p.z - o.z, L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx /= L; dy /= L; dz /= L;
    var b = o.x * dx + o.y * dy + o.z * dz, disc = b * b - (o.x * o.x + o.y * o.y + o.z * o.z - 0.98);
    if (disc < 0) return false;
    var t = -b - Math.sqrt(disc);
    return t > 0 && t < L;
  }
  function updateLabels() {
    var w = viewEl.clientWidth, h = viewEl.clientHeight;
    for (var i = 0; i < labels.length; i++) {
      var l = labels[i];
      tmpV.copy(l.pos).project(camera);
      var show = tmpV.z < 1 && !hiddenByEarth(l.pos);
      if (show !== l.shown) { l.el.style.visibility = show ? "visible" : "hidden"; l.shown = show; }
      if (show) l.el.style.transform = "translate(" + ((tmpV.x * 0.5 + 0.5) * w).toFixed(1) + "px," + ((-tmpV.y * 0.5 + 0.5) * h - l.lift).toFixed(1) + "px) translate(-50%,-100%)";
    }
  }
  function resize() {
    var w = viewEl.clientWidth, h = viewEl.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    if (!cam.zoomed) cam.radius = fitRadius();
  }

  function tick(dt) {
    if (opts.orbits) orbitTime += dt;
    animTime += dt;
    updateSats(dt);
    linkTimer -= dt; if (linkTimer <= 0) { refreshLinks(); linkTimer = 0.25; }
    stepScheduler();
    updatePackets(dt); updateEffects(dt); updateLines();
    if (chipTimer > 0) { chipTimer -= dt; if (chipTimer <= 0) clearChips(); }
  }
  function draw() { updateCamera(); renderer.render(scene, camera); updateLabels(); }
  function frame(ms) {
    requestAnimationFrame(frame);
    var dt = Math.min(0.1, (ms - lastFrame) / 1000 || 0); lastFrame = ms;
    if (!visible) return;
    tick(dt); draw();
  }
  // For tests and debugging: advance the view by some seconds without waiting.
  window.Viz3D._advance = function (seconds) {
    for (var k = 0; k < Math.round(seconds * 60); k++) tick(1 / 60);
    draw();
    return { tau: world.tau, created: world.created, scheduled: world.scheduled, rejected: world.rejected, flying: packets.length };
  };

  // ── Wiring ─────────────────────────────────────────────────────────────
  function wire() {
    var canvas = renderer.domElement, drag = null;
    canvas.addEventListener("pointerdown", function (e) { drag = { x: e.clientX, y: e.clientY }; try { canvas.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } canvas.classList.add("dragging"); });
    canvas.addEventListener("pointermove", function (e) {
      if (!drag) return;
      cam.theta += (e.clientX - drag.x) * 0.006;
      cam.phi = Math.min(Math.PI - 0.12, Math.max(0.12, cam.phi - (e.clientY - drag.y) * 0.006));
      drag.x = e.clientX; drag.y = e.clientY;
    });
    function stop() { drag = null; canvas.classList.remove("dragging"); }
    canvas.addEventListener("pointerup", stop); canvas.addEventListener("pointercancel", stop);
    // Plain scrolling must keep scrolling the page: zoom needs Ctrl/Cmd, or full screen.
    canvas.addEventListener("wheel", function (e) {
      if (!(e.ctrlKey || e.metaKey || document.fullscreenElement)) return;
      e.preventDefault(); zoom(e.deltaY > 0 ? 1.1 : 0.9);
    }, { passive: false });
    canvas.addEventListener("keydown", function (e) {
      var k = e.key, used = true;
      if (k === "ArrowLeft") cam.theta += 0.12; else if (k === "ArrowRight") cam.theta -= 0.12;
      else if (k === "ArrowUp") cam.phi = Math.max(0.12, cam.phi - 0.1); else if (k === "ArrowDown") cam.phi = Math.min(Math.PI - 0.12, cam.phi + 0.1);
      else if (k === "+" || k === "=") zoom(0.9); else if (k === "-") zoom(1.1); else used = false;
      if (used) e.preventDefault();
    });

    $("v3-create").addEventListener("click", createTask);
    $("v3-create-quick").addEventListener("click", createTask);
    $("v3-create5").addEventListener("click", function () {
      var token = worldToken;
      for (var i = 0; i < 5; i++) setTimeout(function () { if (token === worldToken) createTask(); }, i * 170);
    });
    $("v3-step").addEventListener("click", function () { decodeRound(); });
    $("v3-run").addEventListener("click", function () { runRequested = true; nextRoundAt = animTime; });
    $("v3-new").addEventListener("click", function () { resetWorld(world.seed + 1); });
    $("v3-world").addEventListener("change", function () { resetWorld(world.seed); });
    $("v3-auto").addEventListener("change", function () { opts.auto = this.checked; runRequested = false; renderStats(); });
    $("v3-ef").addEventListener("change", function () { opts.earliestFree = this.checked; });
    $("v3-orbits").addEventListener("change", function () { opts.orbits = this.checked; });
    var sel = $("v3-preset"), input = $("v3-expr");
    sel.addEventListener("change", function () { input.value = sel.value; setFormula(input.value); });
    input.addEventListener("change", function () { setFormula(input.value); });
    input.addEventListener("keydown", function (e) { if (e.key === "Enter") setFormula(input.value); });
    $("v3-zoom-in").addEventListener("click", function () { zoom(0.85); });
    $("v3-zoom-out").addEventListener("click", function () { zoom(1.18); });
    $("v3-reset-view").addEventListener("click", resetView);
    var stage = $("v3-stage"), full = $("v3-full");
    if (!stage.requestFullscreen) full.hidden = true;
    full.addEventListener("click", function () { if (document.fullscreenElement) document.exitFullscreen(); else stage.requestFullscreen(); });
    document.addEventListener("fullscreenchange", function () { full.textContent = document.fullscreenElement ? "Exit full screen" : "Full screen"; resize(); });

    if (window.ResizeObserver) new ResizeObserver(resize).observe(viewEl); else window.addEventListener("resize", resize);
    if (window.IntersectionObserver) new IntersectionObserver(function (es) { visible = es[es.length - 1].isIntersecting; }).observe(viewEl);
  }

  function fail(text) {
    var m = $("v3-msg"); m.textContent = text; m.hidden = false;
    ["v3-create", "v3-create-quick", "v3-create5", "v3-step", "v3-run", "v3-new"].forEach(function (id) { $(id).disabled = true; });
  }

  function init() {
    viewEl = $("v3-view"); labelsEl = $("v3-labels");
    if (!viewEl) return;
    THREE = window.THREE;
    if (!THREE) { fail("The 3D view needs the three.js library, which is loaded from the internet (cdnjs). Connect and reload the page. The decoder playground on the Decoder & fitness page works offline."); return; }
    try { renderer = new THREE.WebGLRenderer({ canvas: $("v3-canvas"), antialias: true }); }
    catch (e) { fail("This browser could not start WebGL, which the 3D view needs. Try another browser, or use the decoder playground on the Decoder & fitness page."); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x05070d, 1);
    tmpV = new THREE.Vector3();

    opts.orbits = !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    $("v3-orbits").checked = opts.orbits; $("v3-auto").checked = true; $("v3-ef").checked = false;
    buildScene();
    updateSats(0); refreshLinks();
    $("v3-expr").value = $("v3-preset").value; setFormula($("v3-expr").value);
    wire(); resize(); resetView();
    resetWorld(1);
    requestAnimationFrame(frame);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
