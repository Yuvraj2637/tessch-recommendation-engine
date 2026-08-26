/**
 * Phase 1 — dwell-time tracking.
 *
 * Measures how long a product page is actually being looked at, then sends that
 * one number to the backend when the visitor leaves.
 *
 * Three things here are less obvious than the plan makes them sound, and each one
 * is the difference between usable data and garbage:
 *
 * 1. THE TIMER PAUSES WHEN THE PAGE IS HIDDEN. A naive `Date.now()` at load and
 *    another at unload measures how long the *tab existed*, not how long the
 *    product was looked at. Someone who opens six products in background tabs and
 *    reads them an hour later would register six hour-long "views". This tracker
 *    accumulates only visible time, via `visibilitychange`. The server-side clamp
 *    at 1800s is a backstop for the cases this misses; it is not the main defence.
 *
 * 2. IT SENDS ON `pagehide`, NOT `unload`. `unload` never fires on iOS Safari and
 *    it disables the back/forward cache where it does fire, so a chunk of real
 *    visits would silently record nothing. `pagehide` fires reliably on both
 *    navigation and tab close.
 *
 * 3. IT USES `navigator.sendBeacon`. A normal `fetch` during page teardown gets
 *    cancelled — the browser is already discarding the document. `sendBeacon`
 *    hands the request to the browser to deliver independently of the page. There
 *    is a `fetch(..., {keepalive: true})` fallback for the rare browser without
 *    it.
 *
 * The anonymous id is a random value in `localStorage`. No login, no cookie, no
 * fingerprinting: it identifies a browser, it is not tied to a person, and
 * clearing site data ends it. It is generated to match the `anon_<hex>` format
 * that `backend/services/ingest.py` validates — change one and you must change
 * both.
 */
(function (window, document) {
  "use strict";

  var STORAGE_KEY = "tessch_visitor_id";
  var ENDPOINT = "/api/events";

  /** Matches ANON_ID_PATTERN in backend/services/ingest.py. */
  var ID_PREFIX = "anon_";
  var ID_HEX_CHARS = 12;

  /**
   * Views shorter than this are not sent at all.
   *
   * Mirrors MIN_DWELL_SECONDS on the server. Filtering here as well is not
   * redundant: it avoids a request per mis-click, so a visitor clicking quickly
   * through a grid does not generate a burst of rejections that look like errors
   * in the server log.
   */
  var MIN_SECONDS = 1;

  /**
   * Long visible stretches are flushed periodically rather than held until the
   * page closes. Without this, a genuinely long reading session that ends in a
   * browser crash or a force-quit records nothing at all.
   */
  var FLUSH_INTERVAL_MS = 120000;

  // ── Visitor identity ──────────────────────────────────────────────────────
  function randomHex(chars) {
    var out = "";
    if (window.crypto && window.crypto.getRandomValues) {
      var bytes = new Uint8Array(Math.ceil(chars / 2));
      window.crypto.getRandomValues(bytes);
      for (var i = 0; i < bytes.length; i++) {
        out += ("0" + bytes[i].toString(16)).slice(-2);
      }
    } else {
      // Only reached on browsers without WebCrypto. Math.random is not
      // cryptographically strong, but this id is a bucket label, not a secret —
      // it grants no access and protects nothing.
      while (out.length < chars) {
        out += Math.floor(Math.random() * 16).toString(16);
      }
    }
    return out.slice(0, chars);
  }

  function visitorId() {
    var stored = null;
    try {
      stored = window.localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      // Private browsing or blocked storage. Fall through to a per-page id: the
      // visitor gets no personalisation, which is the correct outcome when we
      // cannot remember them, and nothing breaks.
      stored = null;
    }

    if (stored && /^anon_[0-9a-f]{8,16}$/.test(stored)) {
      return stored;
    }

    var fresh = ID_PREFIX + randomHex(ID_HEX_CHARS);
    try {
      window.localStorage.setItem(STORAGE_KEY, fresh);
    } catch (err) {
      /* not persistable — see above */
    }
    return fresh;
  }

  function resetVisitor() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      /* nothing to clear */
    }
    return visitorId();
  }

  // ── Dwell accumulation ────────────────────────────────────────────────────
  function Tracker() {
    this.productId = null;
    this.surface = null;
    this.position = null;
    this.accumulatedMs = 0;
    this.visibleSince = null;
    this.sent = false;
    this.flushTimer = null;
    this.listeners = [];
    this.lastSent = null;
  }

  /** Seconds of *visible* time on the current product so far. */
  Tracker.prototype.elapsedSeconds = function () {
    var live = this.visibleSince ? Date.now() - this.visibleSince : 0;
    return (this.accumulatedMs + live) / 1000;
  };

  Tracker.prototype.isRunning = function () {
    return this.productId !== null && this.visibleSince !== null;
  };

  Tracker.prototype.start = function (productId, options) {
    var opts = options || {};
    this.productId = productId;
    this.surface = opts.surface || "product_page";
    this.position = typeof opts.position === "number" ? opts.position : null;
    this.accumulatedMs = 0;
    this.visibleSince = document.visibilityState === "hidden" ? null : Date.now();
    this.sent = false;

    var self = this;

    this._bind(document, "visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        self.pause();
        // A hidden page may never fire `pagehide` — on mobile the OS can discard
        // a backgrounded tab outright. Flushing here is the only reliable moment.
        self.flush("hidden");
      } else {
        self.resume();
      }
    });

    // `pagehide` covers both navigating away and closing the tab.
    this._bind(window, "pagehide", function () {
      self.pause();
      self.flush("pagehide");
    });

    // Same-page anchor clicks to another product: flush before the navigation so
    // the current view is attributed to the current product.
    this._bind(window, "beforeunload", function () {
      self.pause();
      self.flush("beforeunload");
    });

    this.flushTimer = window.setInterval(function () {
      if (self.elapsedSeconds() >= FLUSH_INTERVAL_MS / 1000) {
        self.flush("interval");
      }
    }, FLUSH_INTERVAL_MS);

    this._emit("start", { productId: productId, seconds: 0 });
    return this;
  };

  Tracker.prototype.pause = function () {
    if (this.visibleSince !== null) {
      this.accumulatedMs += Date.now() - this.visibleSince;
      this.visibleSince = null;
      this._emit("pause", { seconds: this.elapsedSeconds() });
    }
  };

  Tracker.prototype.resume = function () {
    if (this.productId !== null && this.visibleSince === null && !this.sent) {
      this.visibleSince = Date.now();
      this._emit("resume", { seconds: this.elapsedSeconds() });
    }
  };

  /**
   * Send the accumulated dwell time.
   *
   * `sent` guards against double delivery: `visibilitychange` and `pagehide` both
   * fire when a tab is closed, and without the guard every close would record the
   * view twice — doubling that product's apparent dwell time in a way no
   * validation rule could detect.
   *
   * After an interval flush the counter resets and tracking continues, so a long
   * session produces several honest events rather than one unbounded one.
   */
  Tracker.prototype.flush = function (reason) {
    if (this.sent || this.productId === null) {
      return false;
    }

    var seconds = this.elapsedSeconds();
    if (seconds < MIN_SECONDS) {
      return false;
    }

    var payload = {
      user_id: visitorId(),
      product_id: this.productId,
      duration_seconds: Math.round(seconds * 10) / 10,
      timestamp: new Date().toISOString(),
      surface: this.surface,
    };
    if (this.position !== null) {
      payload.position = this.position;
    }

    var delivered = send(payload);
    this.lastSent = {
      productId: payload.product_id,
      seconds: payload.duration_seconds,
      reason: reason || "manual",
      delivered: delivered,
    };
    this._emit("flush", this.lastSent);

    if (reason === "interval") {
      // Keep measuring; the time just reported must not be counted again.
      this.accumulatedMs = 0;
      this.visibleSince = document.visibilityState === "hidden" ? null : Date.now();
    } else {
      this.sent = true;
      window.clearInterval(this.flushTimer);
    }
    return delivered;
  };

  Tracker.prototype._bind = function (target, type, handler) {
    target.addEventListener(type, handler);
    this.listeners.push([target, type, handler]);
  };

  Tracker.prototype.on = function (event, handler) {
    this._handlers = this._handlers || {};
    (this._handlers[event] = this._handlers[event] || []).push(handler);
    return this;
  };

  Tracker.prototype._emit = function (event, detail) {
    var handlers = (this._handlers || {})[event] || [];
    for (var i = 0; i < handlers.length; i++) {
      try {
        handlers[i](detail);
      } catch (err) {
        // A broken UI listener must never stop the measurement it observes.
        if (window.console) window.console.error("[tracker] listener failed", err);
      }
    }
  };

  // ── Delivery ──────────────────────────────────────────────────────────────
  function send(payload) {
    var body = JSON.stringify(payload);

    if (navigator.sendBeacon) {
      try {
        // Typed as application/json so the backend parses it as JSON. A plain
        // string would arrive as text/plain and be rejected.
        var blob = new Blob([body], { type: "application/json" });
        if (navigator.sendBeacon(ENDPOINT, blob)) {
          return true;
        }
      } catch (err) {
        /* fall through to fetch */
      }
    }

    // `keepalive` asks the browser to finish this request even as the document
    // goes away — the same guarantee sendBeacon gives, without the queueing.
    try {
      window.fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        keepalive: true,
      });
      return true;
    } catch (err) {
      if (window.console) window.console.warn("[tracker] delivery failed", err);
      return false;
    }
  }

  window.TesschTracker = {
    visitorId: visitorId,
    resetVisitor: resetVisitor,
    create: function () {
      return new Tracker();
    },
    MIN_SECONDS: MIN_SECONDS,
  };
})(window, document);
