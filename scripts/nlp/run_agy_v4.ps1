param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "VIA_FRAMES_BEFORE", "VIA_FRAMES_AFTER", "ANCHOR_NEAR_FRAMES",
        "GOAL_FRAMES", "DISTRACTOR_FRAMES", "REDIRECT_PREFIX",
        "north_most", "south_most", "west_most", "east_most"
    )]
    [string]$Bank,
    [int]$Count = 12,
    [string]$Output = "D:\phenikaa\results\nlp_v4_candidates\agy_raw.txt",
    [string]$Model = "gemini-3.8-flash-low"
)

$ErrorActionPreference = "Stop"
$rules = @{
    VIA_FRAMES_BEFORE = "use {W}, optionally {I2}; the phrase precedes the final-goal clause and clearly requires visiting W first"
    VIA_FRAMES_AFTER = "use {W}, optionally {I2}; the phrase follows the goal clause and clearly requires visiting W before delivery"
    ANCHOR_NEAR_FRAMES = "use both {G} and {A}; mean the location closest to anchor A"
    GOAL_FRAMES = "use {L}; optionally use {V}, {I}, {T}, {tail}; L is the final destination"
    DISTRACTOR_FRAMES = "use {X}; explicitly reject X as a destination or stop"
    REDIRECT_PREFIX = "use {X}; cancel or replace old destination X and end so a goal clause can follow"
    north_most = "no placeholders; unambiguously mean the northernmost or topmost map position"
    south_most = "no placeholders; unambiguously mean the southernmost or bottommost map position"
    west_most = "no placeholders; unambiguously mean the westernmost or leftmost map position"
    east_most = "no placeholders; unambiguously mean the easternmost or rightmost map position"
}

$prompt = @"
Generate exactly $Count diverse Vietnamese phrases for bank $Bank.
Constraint: $($rules[$Bank]).
Use lowercase ASCII Vietnamese without diacritics. Use natural language, not explanations.
Do not read files, use tools, inspect test data, emit JSON, markdown, numbering, or comments.
Return exactly one phrase per line and nothing else.
"@

New-Item -ItemType Directory -Path (Split-Path $Output) -Force | Out-Null
$result = & agy --mode plan --disable-slash-commands --model $Model --effort low --print=$prompt
if ($LASTEXITCODE -ne 0) { throw "agy failed with exit code $LASTEXITCODE" }
$result | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Host "Saved raw phrases to $Output. No project source was changed." -ForegroundColor Green
