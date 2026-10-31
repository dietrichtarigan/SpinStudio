<#
  Start SpinStudio on a GPU node and open it in the browser.

  Usage (PowerShell):
    .\SpinStudio.ps1                 # quasi08, port 8765
    .\SpinStudio.ps1 -Node quasi09
    .\SpinStudio.ps1 -NoSync         # do not upload local code first

  The server runs on the node; an SSH tunnel forwards it to http://localhost:<Port>.
  Close this window (or press Ctrl+C) to stop the server.
#>
param(
    [string]$Node = "quasi08",
    [int]$Port = 8765,
    [switch]$NoSync
)
$ErrorActionPreference = "Stop"
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$scp = "C:\Windows\System32\OpenSSH\scp.exe"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = "http://localhost:$Port"

Write-Host "SpinStudio -> $Node" -ForegroundColor Cyan
if (-not $NoSync) {
    Write-Host "Uploading code..."
    Get-ChildItem -Path "$root\spinstudio" -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
    & $ssh $Node "rm -rf ~/micromagnetics/spinstudio.new && mkdir -p ~/micromagnetics/spinstudio.new"
    & $scp -r -q "$root\spinstudio\*" "${Node}:micromagnetics/spinstudio.new/"
    & $ssh $Node "cd ~/micromagnetics && rm -rf spinstudio && mv spinstudio.new spinstudio"
}

# open the browser once the tunnel answers
$opener = {
    param($u)
    for ($i = 0; $i -lt 120; $i++) {
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "$u/api/info" | Out-Null; Start-Process $u; return }
        catch { Start-Sleep -Milliseconds 700 }
    }
}
Start-Job -ScriptBlock $opener -ArgumentList $url | Out-Null

Write-Host "Starting server on $Node (port $Port). Browser opens at $url" -ForegroundColor Green
Write-Host "Keep this window open. Ctrl+C stops the server." -ForegroundColor Yellow
& $ssh -t -o ExitOnForwardFailure=yes -L "${Port}:127.0.0.1:${Port}" $Node `
    "cd ~/micromagnetics && exec .venv/bin/python -m spinstudio.server --port $Port"
