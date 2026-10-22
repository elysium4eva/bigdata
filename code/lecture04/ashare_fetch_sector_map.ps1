# fetch-sw-sector-map.ps1 -- build the Shenwan (申万一级行业) sector cache for the stock-heatmap script.
# The upstream script fetches the same endpoints with Python's urllib, which this network
# rejects; PowerShell with a browser user agent works, so this rebuilds the cache file:
#   { "ts": <epoch>, "map": { "<6-digit code>": "<industry>" }, "chg": { "<industry>": <fraction> } }
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
$hdr = @{ 'User-Agent' = $ua; 'Referer' = 'https://www.swsresearch.com/'; 'Accept' = 'application/json, text/plain, */*' }

function Get-Json([string]$Url) {
  # fetch a URL, decode the raw bytes as UTF-8, and return the parsed JSON
  for ($i = 1; $i -le 4; $i++) {
    try {
      $r = Invoke-WebRequest -Uri $Url -Headers $hdr -TimeoutSec 40 -UseBasicParsing
      $bytes = $r.RawContentStream.ToArray()
      $txt = [System.Text.Encoding]::UTF8.GetString($bytes)
      return ($txt | ConvertFrom-Json)
    } catch {
      Start-Sleep -Milliseconds (600 * $i)
    }
  }
  return $null
}

# ---- 1) the 31 level-1 industries and their changes -------------------------
$listUrl = 'https://www.swsresearch.com/institute-sw/api/index_publish/current/?indextype=%E4%B8%80%E7%BA%A7%E8%A1%8C%E4%B8%9A&page=1&page_size=100'
$list = Get-Json $listUrl
if (-not $list) { throw 'could not fetch the Shenwan level-1 list' }
$industries = @($list.data.results)
Write-Host ("level-1 industries: {0}" -f $industries.Count)

$chg = @{}
foreach ($ind in $industries) {
  $code = [string]$ind.swindexcode
  $name = [string]$ind.swindexname
  if (-not $code -or -not $name) { continue }
  $change = 0.0
  try {
    $close = [double]$ind.l6            # latest close
    $pre = [double]$ind.l4             # previous close
    if ($pre -ne 0) { $change = ($close - $pre) / $pre }
  } catch { $change = 0.0 }
  $chg[$name] = [math]::Round($change, 6)
}

# ---- 2) component stocks per industry --------------------------------------
$map = @{}
foreach ($ind in $industries) {
  $code = [string]$ind.swindexcode
  $name = [string]$ind.swindexname
  if (-not $code) { continue }
  $u = "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/?swindexcode=$code&page=1&page_size=10000"
  $j = Get-Json $u
  if (-not $j) { Write-Host ("  {0} {1}: fetch failed" -f $code, $name); continue }
  $rows = @($j.data.results)
  $added = 0
  foreach ($row in $rows) {
    $c = [string]$row.stockcode
    if ($c -match '^\d{1,6}$') { $c = $c.PadLeft(6, '0') }
    if ($c -match '^\d{6}$' -and -not $map.ContainsKey($c)) { $map[$c] = $name; $added++ }
  }
  Write-Host ("  {0} {1}: {2} stocks" -f $code, $name, $added)
  Start-Sleep -Milliseconds 200
}

Write-Host ("mapped codes: {0} | industries: {1}" -f $map.Count, $chg.Count)
$payload = [pscustomobject]@{ ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); map = $map; chg = $chg }
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $here 'sector_map.json'), ($payload | ConvertTo-Json -Depth 5 -Compress), $enc)
Write-Host 'wrote sector_map.json'
