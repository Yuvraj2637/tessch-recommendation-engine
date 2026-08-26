/**
 * Rendering — product tiles, cards, and the attention meter.
 *
 * The catalogue's `image_url` values point at files that were never created
 * (`/images/products/T001.jpg`), so rather than ship broken images or grey boxes,
 * each product is drawn as a "module stack": three offset bands standing in for
 * upper, midsole and outsole, tinted from the product's own colour name. TESSCH
 * sells modular footwear, so a parts diagram is truer to the brand than stock
 * photography would be — and it means every one of the 100 products has a
 * distinct, deterministic visual with no assets to manage.
 *
 * All text insertion goes through `textContent` or `document.createElement`, never
 * `innerHTML` with interpolated data. Product names and descriptions come from the
 * database, and building HTML by string concatenation is how a stray quote in a
 * product name becomes a broken page — or worse.
 */
(function (window, document) {
  "use strict";

  /**
   * Colour name to hex, covering the 28 colours in the catalogue.
   *
   * The catalogue stores colours as names, not hex, so this bridges the two. The
   * fallback derives a stable hue from the string so a colour added later still
   * gets a consistent tint rather than defaulting to grey.
   */
  var COLOURS = {
    Black: "#22262a",
    Charcoal: "#3b4045",
    Grey: "#8d949a",
    White: "#f2f3ef",
    "Off-White": "#e8e6dd",
    Cream: "#e6dcc4",
    Beige: "#d8cbb2",
    Sand: "#d5c4a3",
    Tan: "#bf9a6d",
    Khaki: "#a89670",
    Brown: "#7a5638",
    Cognac: "#94512a",
    Rust: "#a8502a",
    Burgundy: "#6b2436",
    Red: "#b8322b",
    Coral: "#e4715c",
    Pink: "#dc8fa4",
    Purple: "#6f4a8e",
    Navy: "#26365c",
    Blue: "#2f5f96",
    "Steel Blue": "#5b7d9a",
    "Sky Blue": "#8fbcd8",
    "Ice Blue": "#cfe0e8",
    Teal: "#1f7a74",
    "Dark Green": "#28452f",
    Olive: "#6b7040",
    "Neon Green": "#a8d431",
    Yellow: "#dcbb35",
  };

  function tint(colourName) {
    if (colourName && COLOURS[colourName]) {
      return COLOURS[colourName];
    }
    // Stable pseudo-random hue from the name, so unknown colours are at least
    // consistent between page loads.
    var hash = 0;
    var name = String(colourName || "unknown");
    for (var i = 0; i < name.length; i++) {
      hash = (hash * 31 + name.charCodeAt(i)) % 360;
    }
    return "hsl(" + hash + ", 26%, 48%)";
  }

  /** A very light wash of the product colour, for the tile background. */
  function tileBackground(hex) {
    if (hex.indexOf("#") !== 0) {
      return "#e4e7e0";
    }
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    var mix = function (channel) {
      return Math.round(channel + (233 - channel) * 0.82);
    };
    return "rgb(" + mix(r) + "," + mix(g) + "," + mix(b) + ")";
  }

  function rupees(value) {
    if (value === null || value === undefined) {
      return "—";
    }
    return "₹" + Number(value).toLocaleString("en-IN");
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  /** The module-stack tile. `showCode` puts the part number in the corner. */
  function tile(product, options) {
    var opts = options || {};
    var hue = tint(product.color);
    var node = el("div", "tile");
    node.style.setProperty("--tile-hue", hue);
    node.style.setProperty("--tile-bg", tileBackground(hue));
    node.appendChild(el("i"));
    node.appendChild(el("i"));
    node.appendChild(el("i"));

    if (product.image_url) {
      var img = el("img", "tile__img");
      img.src = product.image_url;
      img.alt = product.product_name || "TESSCH Modular Sneaker";
      img.loading = "lazy";
      img.onerror = function () {
        img.style.display = "none";
      };
      node.appendChild(img);
    }

    if (opts.showCode !== false) {
      node.appendChild(el("span", "tile__code", product.product_id));
    }
    // Only rendered when the backend actually returned a score. The popularity
    // fallback returns null, and showing a fabricated number would hide which
    // strategy produced the list.
    if (typeof opts.score === "number") {
      node.appendChild(el("span", "tile__score", opts.score.toFixed(3)));
    }
    return node;
  }

  /** A product card linking to its detail page. */
  function card(product, options) {
    var opts = options || {};
    var link = el("a", "card");
    link.href = "/product/" + encodeURIComponent(product.product_id);
    link.setAttribute(
      "aria-label",
      product.product_name + ", " + rupees(product.price)
    );
    if (product.availability === false) {
      link.setAttribute("data-unavailable", "true");
    }
    if (typeof opts.index === "number") {
      // Staggered entrance, capped so a 100-item grid does not take 5 seconds.
      link.style.animationDelay = Math.min(opts.index * 28, 420) + "ms";
    }

    link.appendChild(
      tile(product, { score: opts.showScore ? product.score : undefined })
    );

    var body = el("div", "card__body");
    body.appendChild(el("div", "card__name", product.product_name));
    body.appendChild(
      el(
        "div",
        "card__cat",
        [product.category, product.style].filter(Boolean).join(" · ")
      )
    );

    var foot = el("div", "card__foot");
    foot.appendChild(el("span", "card__price", rupees(product.price)));
    if (opts.note) {
      foot.appendChild(el("span", "card__note", opts.note));
    }
    body.appendChild(foot);
    link.appendChild(body);
    return link;
  }

  /** Render a list of products into a container, replacing what was there. */
  function fillGrid(container, products, options) {
    var opts = options || {};
    container.textContent = "";
    if (!products || !products.length) {
      return;
    }
    products.forEach(function (product, index) {
      container.appendChild(
        card(product, {
          index: index,
          showScore: opts.showScore,
          note: opts.note ? opts.note(product) : null,
        })
      );
    });
  }

  /** A labelled strategy badge, so every list says which model produced it. */
  function strategyBadge(node, response) {
    if (!node) return;
    var labels = {
      personal_dwell: "your dwell time",
      item_item_cosine: "item-item cosine",
      popular: "popularity fallback",
      empty: "no data",
    };
    node.textContent = labels[response.strategy] || response.strategy;
    node.setAttribute("data-strategy", response.strategy);
    node.title = response.model_name + " v" + response.model_version;
  }

  function showNote(container, message, variant) {
    container.textContent = "";
    var note = el("div", "note" + (variant ? " note--" + variant : ""));
    note.textContent = message;
    container.appendChild(note);
  }

  function showSkeletons(container, count) {
    container.textContent = "";
    for (var i = 0; i < (count || 4); i++) {
      container.appendChild(el("div", "skeleton"));
    }
  }

  // ── The attention meter ───────────────────────────────────────────────────
  /**
   * The signature element: a pinned instrument showing the dwell timer as it
   * counts, whether it is running, and what has been sent.
   *
   * It exists because the mechanism this project is about is invisible by nature.
   * A reviewer cannot tell whether attention is being measured correctly — or
   * whether the timer pauses on a hidden tab — unless something shows them. This
   * does.
   */
  function attentionMeter(tracker) {
    var root = el("aside", "meter");
    root.setAttribute("data-state", "idle");
    root.setAttribute("aria-live", "off");

    var bar = el("div", "meter__bar");
    bar.appendChild(el("span", "meter__dot"));
    var barLabel = el("span", null, "attention");
    bar.appendChild(barLabel);
    var toggle = el("button", "meter__toggle", "hide");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "true");
    bar.appendChild(toggle);
    root.appendChild(bar);

    var body = el("div", "meter__body");
    body.appendChild(el("div", "meter__label", "time on this page"));
    var clock = el("div", "meter__clock num", "0.0s");
    body.appendChild(clock);
    var status = el("div", "meter__label", "waiting");
    status.style.marginTop = "0.45rem";
    body.appendChild(status);
    var log = el("ul", "meter__log");
    body.appendChild(log);
    root.appendChild(body);

    toggle.addEventListener("click", function () {
      var collapsed = root.getAttribute("data-collapsed") === "true";
      root.setAttribute("data-collapsed", collapsed ? "false" : "true");
      toggle.textContent = collapsed ? "hide" : "show";
      toggle.setAttribute("aria-expanded", collapsed ? "true" : "false");
    });

    function paint() {
      var seconds = tracker.elapsedSeconds();
      clock.textContent = seconds.toFixed(1) + "s";
      var state = tracker.isRunning()
        ? "running"
        : tracker.productId
        ? "paused"
        : "idle";
      root.setAttribute("data-state", state);
      barLabel.textContent =
        state === "running"
          ? "recording"
          : state === "paused"
          ? "paused · tab hidden"
          : "attention";
      if (state === "paused") {
        status.textContent = "paused — hidden tabs are not counted";
      } else if (state === "running") {
        status.textContent = "sends when you leave this page";
      }
    }

    tracker.on("flush", function (detail) {
      var row = el("li");
      row.appendChild(el("b", null, detail.productId));
      row.appendChild(
        el("span", null, detail.seconds.toFixed(1) + "s → " + detail.reason)
      );
      log.insertBefore(row, log.firstChild);
      status.textContent = detail.delivered ? "sent" : "delivery failed";
    });

    // 100ms rather than requestAnimationFrame: this is a numeric readout, not an
    // animation, and repainting 60 times a second to change one decimal place is
    // wasted battery.
    window.setInterval(paint, 100);
    paint();

    document.body.appendChild(root);
    return root;
  }

  window.TesschRender = {
    tint: tint,
    rupees: rupees,
    el: el,
    tile: tile,
    card: card,
    fillGrid: fillGrid,
    strategyBadge: strategyBadge,
    showNote: showNote,
    showSkeletons: showSkeletons,
    attentionMeter: attentionMeter,
  };
})(window, document);
