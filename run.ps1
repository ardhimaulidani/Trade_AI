# ========================================
# Auto Runner for Trade_AI
# Runs every 15 minutes and includes daily UTC trigger for steps 5–6
# ========================================

# --- Configuration ---
$MinutesList = @(0, 15, 30, 45)
$StartSecond = 5
$PythonExe = "python"
$ProjectPath = "D:\Projects\Trade_AI"
Set-Location $ProjectPath

function Run-Step($stepNum, $desc, $cmd) {
    Write-Host ""
    Write-Host "[$stepNum/9] $desc ..."
    & $PythonExe -m $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ ERROR: $cmd failed with code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "✅ [$stepNum/9] $desc completed successfully!"
}

while ($true) {
    $now = Get-Date
    $utcNow = $now.ToUniversalTime()
    Write-Host "`n========================================"
    Write-Host "Starting run at $($now) (UTC: $($utcNow))"
    Write-Host "========================================"

    try {
        # --- [1/9] ---
        Run-Step 1 "Updating Database" "preprocess.db_pull"

        # --- [2/9] ---
        Run-Step 2 "Updating Deriv Database" "preprocess.deriv_pull"

        # --- [3/9] ---
        Run-Step 3 "Building Volume Profile" "preprocess.VP_feature_builder"

        # --- [4/9] ---
        Run-Step 4 "Building Rolling TPO" "preprocess.Rolling_TPO"

        # --- [5–6/9] Daily Steps ---
        if ($utcNow.Hour -eq 0 -and $utcNow.Minute -ge 15 -and $utcNow.Minute -lt 20) {
            Run-Step 5 "Building Daily TPO (00:15 UTC run)" "preprocess.Daily_TPO"
            Run-Step 6 "Building Composite TPO (00:15 UTC run)" "preprocess.Composite_TPO"
        }
        else {
            Write-Host "`n[5/9 + 6/9] Skipped Daily/Composite TPO (not 00:15 UTC)"
        }

        # --- [7/9] ---
        Run-Step 7 "Building Full TPO" "preprocess.TPO_feature_builder"

        # --- [8/9] ---
        Run-Step 8 "Building Derivative" "preprocess.future_feature_builder"

        # --- [9/9] ---
        Run-Step 9 "Starting to detect trades" "main"

        Write-Host "`n✅ All steps completed successfully!"
        Write-Host "========================================`n"
    }
    catch {
        Write-Host "❌ Fatal error: $($_.Exception.Message)"
    }

    # --- Schedule Next Run ---
    $now = Get-Date
    $candidateRuns = @()
    foreach ($minute in $MinutesList) {
        $runTime = Get-Date -Hour $now.Hour -Minute $minute -Second $StartSecond
        if ($runTime -le $now) {
            $runTime = $runTime.AddHours(1)
        }
        $candidateRuns += $runTime
    }

    $nextRun = $candidateRuns | Sort-Object | Select-Object -First 1
    $sleepSeconds = [math]::Max(0, ($nextRun - $now).TotalSeconds)

    Write-Host ("Sleeping {0:N0}s until next run at {1}" -f $sleepSeconds, $nextRun)
    Start-Sleep -Seconds $sleepSeconds
}
