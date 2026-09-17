# publish-via-api.ps1 -- publish the course site to GitHub through the REST API.
# Used because github.com (git over HTTPS) is unreachable from this network while
# api.github.com works.  ASCII-only on purpose (avoids script encoding issues).
#
# Flow: ensure repo -> make the repo non-empty (Contents API creates commit #1)
#       -> upload every file as a blob -> one tree -> commit #2 -> move main
#       -> enable GitHub Pages -> wait until the published URL answers.
param(
  [string]$Owner   = 'elysium4eva',
  [string]$Repo    = 'bigdata',
  [string]$SiteDir = 'D:\google\teaching\big data analytics\lecture notes 2026\website',
  [string]$Message = 'Course site: syllabus, schedule and Lecture 1 courseware',
  [int]$MaxBlobMB  = 60,
  [switch]$SkipPages,
  [switch]$NoWait
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# ---- 1) token from the Windows credential store (never printed) --------------
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'never'
$cred  = "protocol=https`nhost=github.com`n`n" | git credential fill
$token = ($cred | Where-Object { $_ -like 'password=*' }) -replace '^password=', ''
if (-not $token) { throw 'no stored GitHub credential found (run git push once to store one)' }
$user = ($cred | Where-Object { $_ -like 'username=*' }) -replace '^username=', ''
Write-Host "credential found for user: $user"

$api = 'https://api.github.com'
$hdrs = @{
  Authorization = "token $token"
  'User-Agent'  = 'course-site-publisher'
  Accept        = 'application/vnd.github+json'
}

function Invoke-Api {
  param([string]$Method, [string]$Url, $Body, [switch]$AllowFailure)
  $p = @{ Method = $Method; Uri = $Url; Headers = $hdrs; TimeoutSec = 3600 }
  if ($null -ne $Body) { $p['Body'] = $Body; $p['ContentType'] = 'application/json' }
  try { return Invoke-RestMethod @p }
  catch {
    $code = 0; $text = ''
    if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $text = $_.ErrorDetails.Message }
    if (-not $text) { $text = $_.Exception.Message }
    if ($AllowFailure) { return [pscustomobject]@{ __status = $code; __text = $text } }
    throw "API $Method $Url failed (code=$code): $text"
  }
}

# ---- 2) repository ----------------------------------------------------------
$repoInfo = Invoke-Api GET "$api/repos/$Owner/$Repo" $null -AllowFailure
if ($repoInfo.__status) {
  $body = @{
    name        = $Repo
    description = 'Course site for Big Data Analytics (Shenzhen University, 2026 International Business)'
    homepage    = "https://$Owner.github.io/$Repo/"
    private     = $false
    has_issues  = $true
    has_wiki    = $false
    auto_init   = $false
  } | ConvertTo-Json
  Invoke-Api POST "$api/user/repos" $body | Out-Null
  Write-Host "created repository $Owner/$Repo"
  $branch = 'main'
} else {
  $branch = $repoInfo.default_branch
  if (-not $branch) { $branch = 'main' }
  Write-Host "repository $Owner/$Repo exists (default branch: $branch)"
}

# ---- 3) the Git Data API needs a non-empty repo: seed commit #1 -------------
$refInfo = Invoke-Api GET "$api/repos/$Owner/$Repo/git/ref/heads/$branch" $null -AllowFailure
if ($refInfo.__status) {
  $seedPath = Join-Path $SiteDir 'README.md'
  if (-not (Test-Path $seedPath)) { throw "seed file not found: $seedPath" }
  $seedBody = @{
    message = 'Initialise course site repository'
    content = [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($seedPath))
    branch  = $branch
  } | ConvertTo-Json -Compress
  $seed = Invoke-Api PUT "$api/repos/$Owner/$Repo/contents/README.md" $seedBody
  Write-Host ("seeded repository with commit " + $seed.commit.sha)
  $refInfo = Invoke-Api GET "$api/repos/$Owner/$Repo/git/ref/heads/$branch" $null
}
$parentSha = $refInfo.object.sha

# ---- 4) upload every file as a blob ----------------------------------------
$files = Get-ChildItem -Path $SiteDir -Recurse -File |
         Where-Object { $_.FullName -notmatch '\\\.git\\' } |
         Sort-Object FullName
$entries = @()
$totalMB = 0.0
foreach ($f in $files) {
  $rel = $f.FullName.Substring($SiteDir.Length + 1).Replace('\', '/')
  $mb  = [math]::Round($f.Length / 1MB, 2)
  if ($mb -gt $MaxBlobMB) {
    Write-Host ("  SKIP (over {0} MB): {1} ({2} MB)" -f $MaxBlobMB, $rel, $mb)
    continue
  }
  $b64  = [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($f.FullName))
  $body = @{ content = $b64; encoding = 'base64' } | ConvertTo-Json -Compress
  $blob = Invoke-Api POST "$api/repos/$Owner/$Repo/git/blobs" $body
  $entries += @{ path = $rel; mode = '100644'; type = 'blob'; sha = $blob.sha }
  $totalMB += $mb
  Write-Host ("  blob ok: {0} ({1} MB)" -f $rel, $mb)
}
if ($entries.Count -eq 0) { throw 'nothing to upload' }

# ---- 5) tree + commit + move the branch ------------------------------------
$tree = Invoke-Api POST "$api/repos/$Owner/$Repo/git/trees" (@{ tree = $entries } | ConvertTo-Json -Depth 6 -Compress)
$commit = Invoke-Api POST "$api/repos/$Owner/$Repo/git/commits" (@{
  message = $Message; tree = $tree.sha; parents = @($parentSha)
} | ConvertTo-Json)
Write-Host ("commit: " + $commit.sha)
Invoke-Api PATCH "$api/repos/$Owner/$Repo/git/refs/heads/$branch" (@{ sha = $commit.sha; force = $false } | ConvertTo-Json) | Out-Null
Write-Host "branch $branch updated"

# ---- 6) enable GitHub Pages ------------------------------------------------
if (-not $SkipPages) {
  $pagesBody = @{ source = @{ branch = $branch; path = '/' } } | ConvertTo-Json
  $res = Invoke-Api POST "$api/repos/$Owner/$Repo/pages" $pagesBody -AllowFailure
  if ($null -eq $res.__status) { Write-Host "GitHub Pages enabled (branch $branch, root)" }
  elseif ($res.__status -eq 409) { Write-Host 'Pages already enabled' }
  else { Write-Host ("could not enable Pages via API (code=" + $res.__status + "): " + $res.__text) }
}

Write-Host ("uploaded {0} files, {1} MB total" -f $entries.Count, [math]::Round($totalMB, 1))
$url = "https://$Owner.github.io/$Repo/"

# ---- 7) wait until the site answers ---------------------------------------
if (-not $NoWait) {
  Write-Host "waiting for $url ..."
  for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 10
    $res = Invoke-Api GET "https://api.github.com/repos/$Owner/$Repo/pages" $null -AllowFailure
    $status = if ($res.__status) { "api:$($res.__status)" } else { "build:$($res.status)" }
    try {
      $r = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 20 -UseBasicParsing
      if ($r.StatusCode -eq 200) { Write-Host "LIVE: $url (HTTP 200) [$status]"; break }
    } catch {
      $code = 0
      if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
      Write-Host "  attempt $i : http=$code [$status]"
    }
  }
}
Write-Host "done. site: $url"
