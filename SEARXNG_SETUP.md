# SearXNG setup (one-time)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

1. Start the container (this generates a default config on first run):

   ```bash
   docker run -d --name searxng -p 8080:8080 -v "${PWD}/searxng:/etc/searxng" searxng/searxng:latest
   ```

2. Enable the JSON API — SearXNG only serves HTML by default. Open the generated
   `searxng/settings.yml`, find the `search:` section, and make sure `formats`
   includes `json`:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

3. Restart the container to pick up the config change:

   ```bash
   docker restart searxng
   ```

4. Verify it works:

   ```bash
   curl "http://localhost:8080/search?q=test&format=json"
   ```

   You should get back JSON with a `results` array, not an error.

Once that's working, N.O.A.H's web search will work automatically — `server.ts`
already points at `http://localhost:8080` via `SEARXNG_URL` in `.env`. No
restart of the SearXNG container is needed after this point unless you change its config again.
