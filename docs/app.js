import { buildModel, createDwellClock, MAX_DWELL, MIN_DWELL, rankPersonal } from "./model.mjs";

const STORAGE_KEY = "tessch-prototype:activity:v1";
const PAGE_SIZE = 12;
const LIMIT = 4;
const currency = new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 0,
});
const photos = {
  T001: "TANGELO", T002: "SAKURA", T003: "OMNITRIX", T004: "SAHARA", T005: "IKKA",
};
const tones = {
  Black: "#303236", White: "#d2cfc2", Grey: "#828b8f", Blue: "#506d92",
  Navy: "#3b4967", Brown: "#7d5440", Tan: "#a87e60", Red: "#c74b42",
  Pink: "#ce849a", Coral: "#e9785b", Green: "#647754", "Neon Green": "#97ac53",
  Olive: "#707451", Orange: "#ce7546", Beige: "#b39e86", Sand: "#b9a88c",
};

const byId = (id) => document.getElementById(id);
const dialog = byId("detail");
const search = byId("search");
const category = byId("category");
const reset = byId("reset");
const sample = byId("sample");
const more = byId("more");
let catalogue = [];
let products = new Map();
let model;
let history = [];
let visibleCount = PAGE_SIZE;
let current = null;
let clock = null;
let opener = null;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function readActivity() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(raw)) return [];
    return raw.filter((event) => event && products.has(event.id)
      && Number.isFinite(event.seconds) && event.seconds >= MIN_DWELL
      && event.seconds <= MAX_DWELL && Number.isFinite(event.at)).slice(-300);
  } catch {
    return history;
  }
}

function saveActivity(events) {
  history = events.slice(-300);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  } catch {
    // Activity stays available in memory if browser storage is unavailable.
  }
}

function productArt(product, eager = false) {
  const art = node("div", "product-art");
  const tone = tones[product.color] ?? tones[product.color.split(" ")[0]] ?? "#8b8073";
  art.style.setProperty("--tint", tone);
  art.setAttribute("aria-hidden", "true");
  const modules = node("div", "modules");
  modules.append(node("i"), node("i"), node("i"));
  art.append(modules);
  const photo = photos[product.product_id];
  if (photo) {
    const image = node("img", "product-photo");
    image.alt = "";
    image.loading = eager ? "eager" : "lazy";
    image.referrerPolicy = "no-referrer";
    image.addEventListener("load", () => art.classList.add("has-photo"), { once: true });
    image.addEventListener("error", () => image.remove(), { once: true });
    image.src = "https://tessch.in/products/" + photo + ".jpg";
    art.append(image);
  }
  return art;
}

function card(product, note) {
  const button = node("button", "product-card");
  button.type = "button";
  button.dataset.product = product.product_id;
  const body = node("span", "card-body");
  body.append(node("span", "card-category", product.category),
    node("strong", "", product.product_name),
    node("span", "card-price", currency.format(product.price)));
  if (note) body.append(node("span", "card-note", note));
  button.append(productArt(product), body);
  button.addEventListener("click", () => openDetail(product, button));
  return button;
}

function showPicks() {
  const ranked = rankPersonal(history).slice(0, LIMIT);
  const selected = ranked.length ? ranked : model.popular(LIMIT);
  const strategy = byId("pick-strategy");
  strategy.hidden = false;
  strategy.textContent = ranked.length ? "Personal dwell" : "Popular";
  byId("pick-reason").textContent = ranked.length
    ? "Ranked by active viewing time, with more weight on recent visits."
    : "Popular with simulated shoppers. Spend at least one second on a product to personalise this list.";
  byId("picks").replaceChildren(...selected.map(({ id, seconds }) =>
    card(products.get(id), seconds === undefined ? "" : seconds.toFixed(1) + " s viewed")));
  reset.disabled = history.length === 0;
  sample.disabled = history.length > 0;
}

function showCatalogue() {
  const query = search.value.trim().toLowerCase();
  const matching = catalogue.filter((product) =>
    (!category.value || product.category === category.value)
    && [product.product_name, product.category, product.style, product.color]
      .some((value) => value.toLowerCase().includes(query)));
  byId("products").replaceChildren(...matching.slice(0, visibleCount).map((product) => card(product)));
  byId("catalogue-count").textContent = matching.length
    ? "Showing " + Math.min(visibleCount, matching.length) + " of " + matching.length + " products"
    : "No products match your search.";
  more.hidden = matching.length <= visibleCount;
}

function saveDwell() {
  if (!current || !clock) return;
  clock.pause();
  const seconds = clock.seconds();
  clock = null;
  if (seconds >= MIN_DWELL) {
    saveActivity([...readActivity(), { id: current.product_id, seconds, at: Date.now() }]);
    showPicks();
  }
}

function updateTime() {
  if (clock) byId("dwell-time").textContent = clock.seconds().toFixed(1) + " s";
}

function startClock() {
  clock = createDwellClock();
  if (!document.hidden) clock.resume();
  updateTime();
}

function openDetail(product, trigger) {
  if (dialog.open) saveDwell();
  else opener = trigger;

  current = product;
  byId("detail-art").replaceChildren(productArt(product, true));
  byId("detail-title").textContent = product.product_name;
  byId("detail-category").textContent = product.category + " · " + product.collection;
  byId("detail-price").textContent = currency.format(product.price);
  byId("detail-description").textContent = product.description;
  byId("detail-specs").textContent = [product.color, product.material, product.style].join(" · ");

  const similar = model.related(product.product_id, LIMIT);
  const related = similar.length ? similar : model.popular(LIMIT, product.product_id);
  byId("related-strategy").textContent = similar.length ? "Item similarity" : "Popular";
  byId("related-reason").textContent = similar.length
    ? "These products received attention from the same simulated shoppers."
    : "This product has limited browsing data, so these picks are based on overall popularity.";
  byId("related").replaceChildren(...related.map(({ id, score }) =>
    card(products.get(id), score === undefined ? "" : "Cosine " + score.toFixed(2))));

  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
  byId("close-detail").focus();
  startClock();
}

dialog.addEventListener("close", () => {
  saveDwell();
  current = null;
  const activeTrigger = opener?.isConnected ? opener :
    [...document.querySelectorAll("button[data-product]")].find(
      (button) => button.dataset.product === opener?.dataset.product);
  (activeTrigger ?? search).focus({ preventScroll: true });
  opener = null;
});

dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

document.addEventListener("visibilitychange", () => {
  if (!clock) return;
  if (document.hidden) clock.pause();
  else clock.resume();
  updateTime();
});
window.addEventListener("pagehide", saveDwell);
window.addEventListener("pageshow", () => {
  if (!model) return;
  history = readActivity();
  showPicks();
  if (dialog.open && current && !clock) startClock();
});
window.addEventListener("storage", (event) => {
  if (model && event.key === STORAGE_KEY) {
    history = readActivity();
    showPicks();
  }
});
window.setInterval(updateTime, 100);

for (const [control, event] of [[search, "input"], [category, "change"]]) {
  control.addEventListener(event, () => {
    visibleCount = PAGE_SIZE;
    showCatalogue();
  });
}

more.addEventListener("click", () => {
  const firstNew = visibleCount;
  visibleCount += PAGE_SIZE;
  showCatalogue();
  byId("products").children[firstNew]?.focus({ preventScroll: true });
});
reset.addEventListener("click", () => {
  saveActivity([]);
  showPicks();
  sample.focus();
});
sample.addEventListener("click", () => {
  const sampleProducts = catalogue.filter((product) => product.category === "Sneaker").slice(0, 3);
  const sampleAt = Date.now();
  saveActivity(sampleProducts.map(({ product_id }, i) => ({
    id: product_id, seconds: [48, 25, 9][i], at: sampleAt,
  })));
  showPicks();
  byId("pick-reason").textContent = "Sample activity loaded. Open any product to add your own viewing time.";
  byId("picks-title").setAttribute("tabindex", "-1");
  byId("picks-title").focus({ preventScroll: true });
  byId("recommendations").scrollIntoView();
});

try {
  const response = await fetch("./demo-data.json");
  if (!response.ok) throw new Error("Catalogue request failed");
  const data = await response.json();
  catalogue = data.products;
  products = new Map(catalogue.map((product) => [product.product_id, product]));
  model = buildModel(catalogue, data.sessions);
  history = readActivity();
  for (const value of [...new Set(catalogue.map((product) => product.category))].sort()) {
    category.add(new Option(value, value));
  }
  search.disabled = false;
  category.disabled = false;
  showPicks();
  showCatalogue();
} catch (error) {
  byId("pick-reason").textContent = "The demo catalogue could not load. Reload the page to try again.";
  console.error(error);
}
