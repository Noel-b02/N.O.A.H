// Standalone web-search module — split out from server.ts specifically so
// agent.ts can use it without creating a circular import (agent.ts also
// needs server.ts's new /api/agent/* routes to import agent.ts, so agent.ts
// importing back from server.ts would deadlock CommonJS's module-loading
// order: server.ts's top-of-file `require('./agent')` would run before
// server.ts had reached the code that defines webSearch, so agent.ts would
// see it as undefined). A leaf module both server.ts and agent.ts import
// from avoids that entirely, matching how mcp.ts/telegram.ts are already
// leaf modules server.ts imports from, never the reverse.

const SEARXNG_URL = process.env.SEARXNG_URL ?? "http://localhost:8080";

export interface WebSearchResult {
  title: string;
  description: string;
  url: string;
}

// Some upstream engines SearXNG aggregates put highlight markup in their
// snippets — strip it so the model prompt gets plain text.
function stripHtmlTags(text: string): string {
  return text.replace(/<\/?[^>]+>/g, "");
}

export async function webSearch(
  query: string,
  options: { categories?: string; timeRange?: string } = {}
): Promise<WebSearchResult[]> {
  try {
    let url = `${SEARXNG_URL}/search?q=${encodeURIComponent(query)}&format=json`;
    if (options.categories) url += `&categories=${encodeURIComponent(options.categories)}`;
    if (options.timeRange) url += `&time_range=${encodeURIComponent(options.timeRange)}`;

    const res = await fetch(url);

    if (!res.ok) {
      console.error("SearXNG search failed:", res.status, await res.text().catch(() => ""));
      return [];
    }

    const data = await res.json() as { results?: any[] };
    const results = data.results ?? [];

    return results.slice(0, 5).map((r: any) => ({
      title: stripHtmlTags(r.title ?? ""),
      description: stripHtmlTags(r.content ?? ""),
      url: r.url ?? ""
    }));
  } catch (err) {
    console.error("Web search error (is SearXNG running at " + SEARXNG_URL + "?):", err);
    return [];
  }
}
