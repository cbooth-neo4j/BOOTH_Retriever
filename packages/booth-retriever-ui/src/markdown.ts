// Tiny, XSS-safe Markdown renderer for BOOTH answer text.
//
// We render LLM-emitted answers, so the input is untrusted. Rather than
// pull in ``marked`` + ``DOMPurify`` (and all of their attack surface), we
// build the DOM tree node-by-node with ``createElement`` / ``textContent``
// and never touch ``innerHTML``. A malicious ``<script>`` in the LLM
// output therefore renders as visible plain text, not as an executing
// script tag.
//
// Supported subset (deliberately minimal):
//   - Paragraphs (separated by a blank line)
//   - Headings: ``#``, ``##``, ``###`` at start of line
//   - Unordered lists: lines starting with ``- `` or ``* ``
//   - Inline: ``**bold**``, ``*italic*``, ``` `code` ```
//
// Unsupported on purpose: code fences, tables, images, links, raw HTML,
// blockquotes, ordered lists. The LLM prompt forbids these so we keep
// the parser tiny.
//
// Public surface: ``renderMarkdown(text)`` returns a ``DocumentFragment``
// the caller can append directly to any container. Empty / whitespace-only
// input returns an empty fragment.

const HEADING = /^(#{1,3})\s+(.+)$/;
const BULLET = /^[-*]\s+(.+)$/;

/** Render Markdown ``text`` as a DOM fragment. */
export function renderMarkdown(text: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  if (!text || !text.trim()) return frag;

  const blocks = splitBlocks(text);
  for (const block of blocks) {
    const node = renderBlock(block);
    if (node) frag.appendChild(node);
  }
  return frag;
}

// ---------------------------------------------------------------------------
// Block-level
// ---------------------------------------------------------------------------

/**
 * Split the document into logical blocks. A "block" is a heading, a
 * contiguous bullet list, or a paragraph (run of non-blank lines that
 * aren't a heading or list).
 */
function splitBlocks(text: string): string[][] {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks: string[][] = [];
  let current: string[] = [];

  const flush = () => {
    if (current.length > 0) {
      blocks.push(current);
      current = [];
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (line === "") {
      flush();
      continue;
    }
    // Headings always stand alone.
    if (HEADING.test(line)) {
      flush();
      blocks.push([line]);
      continue;
    }
    // Switching between bullet/non-bullet flushes so a paragraph after a
    // list (or vice versa) splits cleanly.
    const isBullet = BULLET.test(line);
    const wasBullet = current.length > 0 && BULLET.test(current[0]!);
    if (current.length > 0 && isBullet !== wasBullet) {
      flush();
    }
    current.push(line);
  }
  flush();
  return blocks;
}

function renderBlock(block: string[]): HTMLElement | null {
  if (block.length === 0) return null;
  const first = block[0]!;
  const headingMatch = HEADING.exec(first);
  if (headingMatch && block.length === 1) {
    const level = headingMatch[1]!.length;
    const tag = (`h${Math.min(level + 2, 6)}`) as
      | "h3"
      | "h4"
      | "h5"
      | "h6";
    const node = document.createElement(tag);
    node.className = `md-h md-h${level}`;
    appendInline(node, headingMatch[2]!);
    return node;
  }

  if (BULLET.test(first)) {
    const ul = document.createElement("ul");
    ul.className = "md-list";
    for (const line of block) {
      const m = BULLET.exec(line);
      if (!m) continue;
      const li = document.createElement("li");
      li.className = "md-list-item";
      appendInline(li, m[1]!);
      ul.appendChild(li);
    }
    return ul;
  }

  // Paragraph: join wrapped lines with a single space (Markdown semantics)
  // unless the line ends with two trailing spaces (hard break, but we
  // already trimmed those) — close enough for our LLM output.
  const p = document.createElement("p");
  p.className = "md-p";
  appendInline(p, block.join(" "));
  return p;
}

// ---------------------------------------------------------------------------
// Inline
// ---------------------------------------------------------------------------

interface Token {
  kind: "text" | "bold" | "italic" | "code";
  value: string;
}

/**
 * Tokenize an inline string. The regex is deliberately greedy-but-anchored
 * so we don't cross paragraph boundaries; nested formatting isn't
 * supported (good enough for our prompt's output).
 */
function tokenizeInline(text: string): Token[] {
  const tokens: Token[] = [];
  // Order matters: ``**bold**`` before ``*italic*``; backtick code is
  // tokenized first so its contents are never interpreted.
  const re =
    /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)|([^*`]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m[1]) tokens.push({ kind: "code", value: m[1].slice(1, -1) });
    else if (m[2]) tokens.push({ kind: "bold", value: m[2].slice(2, -2) });
    else if (m[3]) tokens.push({ kind: "italic", value: m[3].slice(1, -1) });
    else if (m[4]) tokens.push({ kind: "text", value: m[4] });
  }
  // Stray ``*`` or ``\`` characters that didn't form a complete token end
  // up untokenized; surface them as plain text to avoid losing content.
  const consumed = tokens.reduce((acc, t) => {
    switch (t.kind) {
      case "code":
        return acc + t.value.length + 2;
      case "bold":
        return acc + t.value.length + 4;
      case "italic":
        return acc + t.value.length + 2;
      default:
        return acc + t.value.length;
    }
  }, 0);
  if (consumed < text.length) {
    tokens.push({ kind: "text", value: text.slice(consumed) });
  }
  return tokens;
}

function appendInline(parent: HTMLElement, text: string): void {
  for (const token of tokenizeInline(text)) {
    if (token.kind === "text") {
      parent.appendChild(document.createTextNode(token.value));
      continue;
    }
    const tag =
      token.kind === "bold"
        ? "strong"
        : token.kind === "italic"
          ? "em"
          : "code";
    const node = document.createElement(tag);
    node.textContent = token.value;
    parent.appendChild(node);
  }
}
