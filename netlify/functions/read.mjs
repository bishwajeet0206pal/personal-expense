import Anthropic from "@anthropic-ai/sdk";

// Models are tried in order. Set AI_MODEL on Netlify to force one.
const MODELS = [process.env.AI_MODEL, "claude-sonnet-4-5-20250929", "claude-sonnet-4-6", "claude-haiku-4-5"].filter(Boolean);

const PROMPT =
  "These images are an Indian invoice, bill, receipt, or payment confirmation. " +
  'Reply with ONLY valid JSON: {"total": number or null, "merchant": string or null, "date": "YYYY-MM-DD" or null, "category": string or null}. ' +
  "CRITICAL RULES FOR MULTI-TABLE INVOICES & ACCURACY:\n" +
  "1. TABLE SELECTION (CRITICAL):\n" +
  "   - Invoices often have multiple tables (e.g., Primary Goods/Items table vs. Annexure, Tax Breakdown, Fee breakdown, or HSN summary).\n" +
  "   - ALWAYS identify and focus on the PRIMARY table where actual purchased items/products are listed (headers like 'Item Description', 'Particulars', 'Product', 'Qty', 'MRP', 'Total').\n" +
  "   - COMPLETELY IGNORE secondary tables such as 'Annexure', 'Tax Breakdown', 'Nature of charge', 'Surge charge', 'Handling charge', or 'HSN/SAC Summary'.\n" +
  "2. TOTAL AMOUNT:\n" +
  "   - 'total' MUST be the final payable grand total from the PRIMARY item table (or verify with 'Amount in Words', e.g. 'Rupees Two Hundred Fifty Eight Only' -> 258).\n" +
  "   - In a table 'Total' row with multiple columns (e.g., Total | Qty: 1 | Taxable: 245.71 | Taxes | Total: 258.00), the total is the final payable column (258.00). NEVER pick the quantity (1) or tax numbers!\n" +
  "   - NEVER take totals or tax amounts from the Annexure table.\n" +
  "3. date must be YYYY-MM-DD. Indian invoices use DD-MM-YYYY or DD/MM/YYYY.\n" +
  "4. merchant is the brand/store name (e.g., Zepto, Blinkit, Swiggy, Zomato, D-Mart, Apollo Pharmacy, Uber, Amazon).\n" +
  "5. category MUST be one of exactly 6 fixed major categories: 'Groceries', 'Food & Dining', 'Shopping', 'Bills & Utilities', 'Travel & Transport', 'Healthcare & Medical'.";

const ALIASES = [
  [/geddit|zepto/i, "Zepto", "Groceries"],
  [/blink\s*commerce|blinkit|grofers/i, "Blinkit", "Groceries"],
  [/instamart/i, "Swiggy Instamart", "Groceries"],
  [/bundl|swiggy/i, "Swiggy", "Food & Dining"],
  [/zomato|eternal limited/i, "Zomato", "Food & Dining"],
  [/bigbasket|innovative retail/i, "BigBasket", "Groceries"],
  [/dmart|d-mart|avenue supermarts/i, "D-Mart", "Groceries"],
  [/netflix/i, "Netflix", "Bills & Utilities"],
  [/spotify/i, "Spotify", "Bills & Utilities"],
  [/prime\s*video|amazon\s*prime/i, "Amazon Prime", "Bills & Utilities"],
  [/youtube\s*premium/i, "YouTube Premium", "Bills & Utilities"],
  [/hdfc.*credit\s*card|hdfc\s*card/i, "HDFC Credit Card", "Bills & Utilities"],
  [/icici.*credit\s*card|icici\s*card/i, "ICICI Credit Card", "Bills & Utilities"],
  [/sbi\s*card/i, "SBI Card", "Bills & Utilities"],
  [/airtel/i, "Airtel", "Bills & Utilities"],
  [/jio/i, "Jio", "Bills & Utilities"],
  [/uber/i, "Uber", "Travel & Transport"],
  [/ola\s*cabs/i, "Ola", "Travel & Transport"],
  [/amazon/i, "Amazon", "Shopping"],
  [/flipkart/i, "Flipkart", "Shopping"],
  [/apollo|medplus|netmeds|pharmacy/i, "Apollo Pharmacy", "Healthcare & Medical"],
];

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" }
  });

function parseModelJson(text) {
  const s = String(text || "").replace(/```json|```/g, "").trim();
  const a = s.indexOf("{"), b = s.lastIndexOf("}");
  if (a < 0 || b < a) throw new Error("no json");
  return JSON.parse(s.slice(a, b + 1));
}

async function tryGemini(images) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  const parts = images.map((data) => ({
    inlineData: { mimeType: "image/jpeg", data }
  }));
  parts.push({ text: PROMPT });

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts }],
        generationConfig: { responseMimeType: "application/json" }
      })
    }
  );
  if (!res.ok) throw new Error("Gemini HTTP " + res.status);
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  return parseModelJson(text);
}

export default async (req) => {
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);

  let body;
  try { body = await req.json(); } catch { return json({ error: "bad_request" }, 400); }
  const { images } = body || {};
  if (!Array.isArray(images) || images.length < 1 || images.length > 3) return json({ error: "bad_request" }, 400);

  let totalBytes = 0;
  for (const im of images) {
    if (typeof im !== "string" || !im.length || !/^[A-Za-z0-9+/=]+$/.test(im.slice(0, 200))) return json({ error: "bad_request" }, 400);
    totalBytes += im.length;
  }
  if (totalBytes > 8_000_000) return json({ error: "too_big" }, 413);

  // 1. Try Gemini if configured
  if (process.env.GEMINI_API_KEY) {
    try {
      const parsed = await tryGemini(images);
      if (parsed) return formatResponse(parsed, "gemini-1.5-flash");
    } catch (err) {
      console.warn("Gemini attempt failed:", err.message);
    }
  }

  // 2. Try Anthropic if configured
  if (process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_BASE_URL) {
    let client;
    try { client = new Anthropic(); } catch { client = null; }
    if (client) {
      const content = images.map((data) => ({ type: "image", source: { type: "base64", media_type: "image/jpeg", data } }));
      content.push({ type: "text", text: PROMPT });

      for (const model of MODELS) {
        try {
          const msg = await client.messages.create({ model, max_tokens: 450, messages: [{ role: "user", content }] });
          const text = (msg.content || []).filter((c) => c.type === "text").map((c) => c.text).join("");
          const d = parseModelJson(text);
          return formatResponse(d, model);
        } catch (e) {
          const st = e && e.status;
          if (st === 404 || st === 400) continue;
          if (st === 429) return json({ error: "rate_limited" }, 429);
          console.warn("Anthropic error:", e.message);
          break;
        }
      }
    }
  }

  // 3. Fallback when no cloud API key is configured
  return json({
    use_client_ocr: true,
    message: "No server AI API key configured; client-side OCR enabled."
  }, 200);
};

const SIX_CATEGORIES = ["Groceries", "Food & Dining", "Shopping", "Bills & Utilities", "Travel & Transport", "Healthcare & Medical"];

function formatResponse(d, model) {
  let merchant = typeof d.merchant === "string" ? d.merchant.trim() : "";
  let category = typeof d.category === "string" ? d.category.trim() : null;

  for (const [re, name, cat] of ALIASES) {
    if (re.test(merchant) || re.test(JSON.stringify(d))) {
      merchant = name;
      if (!category || !SIX_CATEGORIES.includes(category)) category = cat;
      break;
    }
  }

  if (!SIX_CATEGORIES.includes(category)) {
    category = "Groceries";
  }

  const t = d.total == null ? null : Number(String(d.total).replace(/[^0-9.]/g, ""));

  return json({
    total: Number.isFinite(t) && t > 0 ? t : null,
    merchant: merchant || null,
    date: typeof d.date === "string" ? d.date : null,
    category: category,
    model
  });
}

export const config = { path: "/api/read" };
