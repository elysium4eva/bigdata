# publish.ps1 -- initialise this folder as the GitHub repository `bigdata`
# and push it so that GitHub Pages can serve it at
#     https://elysium4eva.github.io/bigdata/
#
# Usage (run inside the website folder):
#     .\publish.ps1 -Remote "https://github.com/elysium4eva/bigdata.git"
#     .\publish.ps1 -Remote "git@github.com:elysium4eva/bigdata.git" -Message "Update"
param(
  [string]$Remote = 'https://github.com/elysium4eva/bigdata.git',
  [string]$Message = 'Course site: syllabus and courseware',
  [string]$Branch  = 'main',
  [switch]$SkipPush,
  [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw 'git not found. Install Git for Windows first: https://git-scm.com/download/win'
}

if (-not (Test-Path (Join-Path $PSScriptRoot '.git'))) {
  git init -b $Branch | Out-Null
  Write-Host "initialised empty repository on branch $Branch"
}

# configure the remote (add or update)
$existing = git remote 2>$null
if ($existing -contains 'origin') { git remote set-url origin $Remote } else { git remote add origin $Remote }

# make sure a commit identity exists (repo-local fallback derived from the remote)
if (-not (git config user.email)) {
  $owner = 'elysium4eva'
  if ($Remote -match 'github\.com[:/]([^/]+)/') { $owner = $Matches[1] }
  git config user.name  $owner
  git config user.email "$owner@users.noreply.github.com"
  Write-Host "set repo-local git identity to $owner <$owner@users.noreply.github.com>"
  Write-Host '  (change it to your GitHub noreply address if you want commits linked to your account)'
}

git add -A
if ($PrepareOnly) {
  Write-Host 'staged all files (-PrepareOnly). Next: .\publish.ps1'
  exit 0
}

if (git status --porcelain) {
  git commit -m $Message | Out-Null
  Write-Host "committed: $Message"
} else {
  Write-Host 'nothing to commit'
}

if ($SkipPush) {
  Write-Host 'skipped push (-SkipPush)'
} else {
  git push -u origin $Branch
  Write-Host ''
  Write-Host 'pushed. Now enable Pages: Settings -> Pages -> Deploy from a branch -> main / (root)'
  Write-Host 'site: https://elysium4eva.github.io/bigdata/'
}
