const HALF_LIFE_MS = 30 * 24 * 60 * 60 * 1000;

export const MAX_DWELL = 1800;
export const MIN_DWELL = 1;

export function rankPersonal(history, now = Date.now()) {
  const byProduct = new Map();
  for (const { id, seconds, at } of history) {
    const entry = byProduct.get(id) ?? { id, score: 0, seconds: 0, lastSeen: 0 };
    const age = Math.max(0, now - at);
    entry.score += Math.log1p(seconds) * 0.5 ** (age / HALF_LIFE_MS);
    entry.seconds += seconds;
    entry.lastSeen = Math.max(entry.lastSeen, at);
    byProduct.set(id, entry);
  }
  return [...byProduct.values()].sort((a, b) =>
    b.score - a.score || b.lastSeen - a.lastSeen || a.id.localeCompare(b.id));
}

export function buildModel(products, sessions) {
  const ids = products.map((product) => product.product_id);
  const index = new Map(ids.map((id, i) => [id, i]));
  const columns = ids.map(() => new Float64Array(sessions.length));
  const popularity = new Map(ids.map((id) => [id, 0]));

  sessions.forEach((views, visitor) => {
    const scores = new Map();
    for (const [id, seconds] of views) {
      scores.set(id, (scores.get(id) ?? 0) + Math.log1p(seconds));
      if (popularity.has(id)) popularity.set(id, popularity.get(id) + seconds);
    }
    if (scores.size < 2) return;
    const rowNorm = Math.hypot(...scores.values());
    for (const [id, score] of scores) {
      if (index.has(id)) columns[index.get(id)][visitor] = score / rowNorm;
    }
  });

  const norms = columns.map((column) => Math.hypot(...column));
  const rankedPopular = [...ids].sort((a, b) =>
    popularity.get(b) - popularity.get(a) || a.localeCompare(b));
  const neighbours = new Map(ids.map((anchor, i) => {
    const ranked = [];
    for (const [j, id] of ids.entries()) {
      if (i === j || !norms[i] || !norms[j]) continue;
      let dot = 0;
      for (let row = 0; row < sessions.length; row += 1) {
        dot += columns[i][row] * columns[j][row];
      }
      if (dot > 0) ranked.push({ id, score: Math.min(1, dot / (norms[i] * norms[j])) });
    }
    ranked.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    return [anchor, ranked];
  }));

  return {
    popular(limit, exclude) {
      return rankedPopular.filter((id) => id !== exclude).slice(0, limit).map((id) => ({ id }));
    },
    related(id, limit) {
      return (neighbours.get(id) ?? []).slice(0, limit);
    },
  };
}

export function createDwellClock(now = () => performance.now()) {
  let started = null;
  let elapsed = 0;
  return {
    resume() {
      if (started === null) started = now();
    },
    pause() {
      if (started === null) return;
      elapsed += now() - started;
      started = null;
    },
    seconds() {
      return Math.min(MAX_DWELL, (elapsed + (started === null ? 0 : now() - started)) / 1000);
    },
  };
}
