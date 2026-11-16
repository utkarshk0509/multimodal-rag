/**
 * AeroRAG motion layer
 * --------------------------------------------------------------------------
 * Built on Motion (https://motion.dev) — the vanilla-JS edition of Framer
 * Motion, sharing the same spring physics and animation engine.
 *
 * Declarative hooks (add to any element in index.html):
 *
 *   data-animate="fade-up"   Entrance when scrolled into view.
 *                            Variants: fade-up | fade-down | fade | scale |
 *                                      blur | slide-left | slide-right
 *   data-delay="0.2"         Extra delay (seconds) for data-animate.
 *   data-stagger             Children enter one after another when in view.
 *   data-stagger-variant     Variant used for stagger children (default fade-up).
 *   data-split               Word-by-word headline reveal.
 *   data-count="83"          Count-up number (with data-suffix / data-decimals).
 *   data-tilt                3D pointer tilt + glare (fine pointers only).
 *   data-magnetic            Element drifts toward the pointer.
 *   data-press               Springy press-down feedback (delegated, so it
 *                            works on elements created later too).
 *
 * Imperative API for app.js lives on window.AeroMotion. app.js calls it
 * through a guard, so the app keeps working if this module fails to load
 * or the visitor prefers reduced motion.
 */
import { animate, inView, scroll, stagger } from "https://cdn.jsdelivr.net/npm/motion@11.11.13/+esm";

const root = document.documentElement;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

/* ---------- Shared transition presets (Framer-style) ---------- */
export const SPRING = {
  soft: { type: "spring", stiffness: 170, damping: 22, mass: 1 },
  snappy: { type: "spring", stiffness: 420, damping: 30, mass: 0.6 },
  bouncy: { type: "spring", stiffness: 520, damping: 18, mass: 0.7 },
  gentle: { type: "spring", stiffness: 90, damping: 20, mass: 1 },
};
const EASE_OUT = [0.16, 1, 0.3, 1];

/* ---------- Variants: keyframe pairs [from, to] ---------- */
const VARIANTS = {
  "fade-up": { opacity: [0, 1], y: [28, 0] },
  "fade-down": { opacity: [0, 1], y: [-16, 0] },
  fade: { opacity: [0, 1] },
  scale: { opacity: [0, 1], scale: [0.94, 1] },
  blur: { opacity: [0, 1], y: [14, 0], filter: ["blur(10px)", "blur(0px)"] },
  "slide-left": { opacity: [0, 1], x: [-28, 0] },
  "slide-right": { opacity: [0, 1], x: [28, 0] },
};

const ENTER_TRANSITION = { duration: 0.8, ease: EASE_OUT };

function variant(name) {
  return VARIANTS[name] || VARIANTS["fade-up"];
}

/** Hide synchronously, then animate in. Prevents a one-frame flash. */
function reveal(el, name = "fade-up", options = {}) {
  el.style.opacity = "0";
  el.setAttribute("data-shown", "");
  return animate(el, variant(name), { ...ENTER_TRANSITION, ...options });
}

function revealMany(elements, name = "fade-up", options = {}) {
  const list = Array.from(elements);
  if (!list.length) return;
  list.forEach((el) => {
    el.style.opacity = "0";
    el.setAttribute("data-shown", "");
  });
  return animate(list, variant(name), {
    ...ENTER_TRANSITION,
    delay: stagger(options.stagger ?? 0.07, { startDelay: options.delay ?? 0 }),
  });
}

/** Run once when the element first scrolls into view. */
function onceInView(el, callback, amount = 0.2) {
  let done = false;
  inView(
    el,
    () => {
      if (done) return;
      done = true;
      callback();
    },
    { amount }
  );
}

/* ---------- Declarative: entrances ---------- */
function setupEntrances() {
  document.querySelectorAll("[data-animate]").forEach((el) => {
    const delay = parseFloat(el.dataset.delay || "0");
    onceInView(el, () => reveal(el, el.dataset.animate, { delay }));
  });

  document.querySelectorAll("[data-stagger]").forEach((container) => {
    const name = container.dataset.staggerVariant || "fade-up";
    const delay = parseFloat(container.dataset.delay || "0");
    onceInView(container, () => {
      container.setAttribute("data-shown", "");
      revealMany(container.children, name, { delay, stagger: 0.08 });
    }, 0.15);
  });
}

/* ---------- Declarative: split headline ---------- */
function splitWords(el) {
  const label = el.textContent.replace(/\s+/g, " ").trim();
  el.setAttribute("aria-label", label);

  const frag = document.createDocumentFragment();
  const words = [];

  Array.from(el.childNodes).forEach((node) => {
    if (node.nodeType !== Node.TEXT_NODE) {
      frag.append(node.cloneNode(true));
      return;
    }
    node.textContent.split(/(\s+)/).forEach((part) => {
      if (!part) return;
      if (/^\s+$/.test(part)) {
        frag.append(" ");
        return;
      }
      const mask = document.createElement("span");
      mask.className = "split-word";
      mask.setAttribute("aria-hidden", "true");
      const inner = document.createElement("span");
      inner.className = "split-word-inner";
      inner.textContent = part;
      inner.style.transform = "translateY(110%)";
      mask.append(inner);
      frag.append(mask);
      words.push(inner);
    });
  });

  el.replaceChildren(frag);
  return words;
}

function setupSplitHeadlines() {
  document.querySelectorAll("[data-split]").forEach((el) => {
    const words = splitWords(el);
    el.setAttribute("data-shown", "");
    const delay = parseFloat(el.dataset.delay || "0");
    animate(
      words,
      { y: ["110%", "0%"] },
      { duration: 0.9, ease: EASE_OUT, delay: stagger(0.06, { startDelay: delay }) }
    );
  });
}

/* ---------- Declarative: count-up stats ---------- */
function setupCounters() {
  document.querySelectorAll("[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    const prefix = el.dataset.prefix || "";
    const format = (v) => `${prefix}${v.toFixed(decimals)}${suffix}`;

    el.textContent = format(0);
    onceInView(el, () => {
      animate(0, target, {
        duration: 1.8,
        ease: EASE_OUT,
        onUpdate: (v) => (el.textContent = format(v)),
      });
    }, 0.6);
  });
}

/* ---------- Declarative: tilt + glare ---------- */
function setupTilt() {
  if (!finePointer) return;
  document.querySelectorAll("[data-tilt]").forEach((el) => {
    const max = parseFloat(el.dataset.tilt || "6");
    animate(el, { transformPerspective: 900 }, { duration: 0 });

    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width;
      const py = (e.clientY - r.top) / r.height;
      el.style.setProperty("--glare-x", `${px * 100}%`);
      el.style.setProperty("--glare-y", `${py * 100}%`);
      animate(
        el,
        { rotateX: (0.5 - py) * max, rotateY: (px - 0.5) * max, y: -4 },
        SPRING.soft
      );
    });

    el.addEventListener("pointerleave", () => {
      animate(el, { rotateX: 0, rotateY: 0, y: 0 }, SPRING.gentle);
    });
  });
}

/* ---------- Declarative: magnetic ---------- */
function setupMagnetic() {
  if (!finePointer) return;
  document.querySelectorAll("[data-magnetic]").forEach((el) => {
    const strength = parseFloat(el.dataset.magnetic || "0.25");
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      animate(el, { x: dx * strength, y: dy * strength }, SPRING.snappy);
    });
    el.addEventListener("pointerleave", () => {
      animate(el, { x: 0, y: 0 }, SPRING.bouncy);
    });
  });
}

/* ---------- Declarative: press feedback (delegated) ---------- */
function setupPress() {
  let pressed = null;
  const release = () => {
    if (!pressed) return;
    animate(pressed, { scale: 1 }, SPRING.bouncy);
    pressed = null;
  };
  document.addEventListener("pointerdown", (e) => {
    const el = e.target.closest("[data-press]");
    if (!el || el.disabled) return;
    pressed = el;
    animate(el, { scale: parseFloat(el.dataset.press || "0.95") }, SPRING.snappy);
  });
  document.addEventListener("pointerup", release);
  document.addEventListener("pointercancel", release);
  window.addEventListener("blur", release);
}

/* ---------- Scroll-linked: progress bar, parallax, navbar ---------- */
function setupScrollEffects() {
  const bar = document.querySelector(".scroll-progress");
  const backdrop = document.querySelector(".ambient-backdrop");
  const navbar = document.querySelector(".navbar");

  scroll((progress) => {
    if (bar) bar.style.transform = `scaleX(${progress})`;
    if (backdrop) backdrop.style.transform = `translate3d(0, ${progress * -140}px, 0)`;
  });

  if (navbar) {
    const update = () => navbar.classList.toggle("is-scrolled", window.scrollY > 24);
    update();
    window.addEventListener("scroll", update, { passive: true });
  }

  // Slow ambient drift on the light blobs
  document.querySelectorAll(".ambient-light").forEach((light, i) => {
    animate(
      light,
      { x: [0, i ? -60 : 70, 0], y: [0, i ? 50 : 40, 0], scale: [1, 1.08, 1] },
      { duration: 18 + i * 4, repeat: Infinity, ease: "easeInOut" }
    );
  });
}

/* ---------- Imperative API used by app.js ---------- */
const AeroMotion = {
  SPRING,

  /** Animate a freshly inserted / revealed element. */
  enter(el, name = "fade-up", options = {}) {
    if (!el) return;
    return reveal(el, name, { ...SPRING.soft, ...options });
  },

  /** Animate out; resolves when finished so the caller can remove the node. */
  exit(el, name = "fade-down") {
    if (!el) return Promise.resolve();
    const v = variant(name);
    const reversed = Object.fromEntries(Object.entries(v).map(([k, [a, b]]) => [k, [b, a]]));
    return animate(el, reversed, { duration: 0.25, ease: "easeIn" }).finished;
  },

  /** Stagger a group of elements in. */
  stagger(elements, name = "fade-up", options = {}) {
    return revealMany(elements, name, { stagger: 0.05, ...options });
  },

  /** Chat bubbles: user slides from the right, assistant from the left. */
  message(row, role) {
    if (!row) return;
    const fromX = role === "user" ? 24 : -24;
    row.style.opacity = "0";
    animate(row, { opacity: [0, 1], x: [fromX, 0], y: [10, 0] }, SPRING.soft);
    const avatar = row.querySelector(".avatar");
    if (avatar) animate(avatar, { scale: [0.4, 1], rotate: [-20, 0] }, { ...SPRING.bouncy, delay: 0.05 });
  },

  /** Toast notifications spring up from below. */
  toastIn(el) {
    if (!el) return;
    el.style.opacity = "0";
    animate(el, { opacity: [0, 1], y: [24, 0], scale: [0.92, 1] }, SPRING.bouncy);
  },

  toastOut(el) {
    if (!el) return Promise.resolve();
    return animate(el, { opacity: 0, x: 40, scale: 0.96 }, { duration: 0.25, ease: "easeIn" }).finished;
  },

  /** Looping "typing" dots; returns a stop() function. */
  thinking(container) {
    const dots = container ? container.querySelectorAll(".typing-dots i") : [];
    if (!dots.length) return () => {};
    const controls = animate(
      dots,
      { y: [0, -5, 0], opacity: [0.35, 1, 0.35] },
      { duration: 0.9, repeat: Infinity, ease: "easeInOut", delay: stagger(0.15) }
    );
    return () => controls.stop();
  },

  /** Success pop. */
  pop(el) {
    if (!el) return;
    animate(el, { scale: [1, 1.06, 1] }, { duration: 0.45, ease: EASE_OUT });
  },

  /** Error shake. */
  shake(el) {
    if (!el) return;
    animate(el, { x: [0, -10, 9, -6, 4, 0] }, { duration: 0.45, ease: "easeOut" });
  },
};

/* ---------- Boot ---------- */
if (!reducedMotion) {
  window.AeroMotion = AeroMotion;
  setupSplitHeadlines();
  setupCounters();
  setupEntrances();
  setupTilt();
  setupMagnetic();
  setupPress();
  setupScrollEffects();
}

root.classList.remove("motion-pending");
root.classList.add(reducedMotion ? "motion-reduced" : "motion-ready");
