import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Konfiguration: Wie wird der Server gestartet?
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"] # Startet unser Server-Skript von oben
)

async def main():
    # 1. Verbindung zum Server via stdio aufbauen
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            
            # 2. Verbindung initialisieren
            await session.initialize()
            print("🤖 Client: Verbindung zum MCP-Server erfolgreich hergestellt!\n")

            # 3. Liste der verfügbaren Tools vom Server abfragen
            tools_response = await session.list_tools()
            print("📋 Verfügbare Tools auf dem Server:")
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}")
            
            # 4. Das Tool "hole_wetter" direkt vom Client aus aufrufen
            print("\n🌤️ Rufe Tool 'hole_wetter' für Berlin auf...")
            ergebnis = await session.call_tool(
                name="hole_wetter", 
                arguments={"stadt": "Berlin"}
            )
            
            # 5. Ergebnis anzeigen
            # (In der Realität würde der Client dieses Ergebnis jetzt an das LLM senden)
            print(f" Ergebnis vom Server: {ergebnis.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())



# Alternative zu eigenem client: json config für Apps wie Cline oder Claude Desktop:
# {
#   "mcpServers": {
#     "mein-wetter-server": {
#       "command": "python",
#       "args": ["C:/Pfad/zu/deinem/projekt/server.py"]
#     }
#   }
# }