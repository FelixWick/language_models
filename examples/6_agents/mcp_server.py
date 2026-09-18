# pip install mcp
from mcp.server.mcpserver import MCPServer

# 1. Server initialisieren
mcp = MCPServer("Wetter-Dienst")

# 2. Ein Tool registrieren (Synchron!)
@mcp.tool()
def hole_wetter(stadt: str) -> str:
    """Holt das aktuelle Wetter für eine angegebene Stadt."""
    stadt_clean = stadt.strip().lower()
    
    # Simulierter Wetter-Abruf (hier könnte eine echte API stehen)
    if stadt_clean == "berlin":
        return "In Berlin sind es aktuell 18°C und es ist leicht bewölkt."
    elif stadt_clean == "münchen":
        return "In München scheint die Sonne bei 22°C."
    else:
        return f"Für {stadt} wurden 15°C und wechselhaftes Wetter gemeldet."

# 3. Server starten (lauscht über Standard-Input/Output)
if __name__ == "__main__":
    mcp.run()
