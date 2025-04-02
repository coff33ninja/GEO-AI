# Define color variables for consistent output
$colorTitle = "Cyan"
$colorMetric = "Green"
$colorValue = "Yellow"
$colorError = "Red"
$colorGraph = "Magenta"

# Define possible installation paths for Speedtest CLI
$SpeedtestPaths = @(
    "C:\Program Files (x86)\Ookla\Speedtest\speedtest.exe",   # Official Ookla Speedtest CLI
    "C:\ProgramData\chocolatey\bin\speedtest.exe",            # Chocolatey variant
    "C:\Program Files\SpeedtestCLI\speedtest.exe"             # Potential Winget variant
)

# Select the first available executable
$SpeedtestPath = $SpeedtestPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $SpeedtestPath) {
    Write-Host "[ERROR] Speedtest CLI not found. Attempting to install via available package managers..." -ForegroundColor $colorError

    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "Installing Speedtest CLI via Chocolatey..." -ForegroundColor $colorValue
        choco install speedtest -y
    }
    elseif (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Speedtest CLI via Winget..." -ForegroundColor $colorValue
        winget install -e --id Ookla.Speedtest.CLI
    }
    else {
        Write-Host "No package manager found. Please install Speedtest CLI manually from: https://www.speedtest.net/apps/cli" -ForegroundColor $colorError
        exit 1
    }
    
    Start-Sleep -Seconds 10  # Wait for installation
    $SpeedtestPath = $SpeedtestPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $SpeedtestPath) {
        Write-Host "[ERROR] Speedtest CLI installation failed." -ForegroundColor $colorError
        exit 1
    }
}

# Function to show a spinner while a background job is running
function Show-Spinner {
    param(
        [string]$Message,
        [scriptblock]$Condition
    )
    $spinner = @("|", "/", "-", "\")
    $i = 0
    Write-Host "$Message " -ForegroundColor $colorTitle -NoNewline
    while (& $Condition) {
        Write-Host ("`b" + $spinner[$i % 4]) -ForegroundColor $colorValue -NoNewline
        Start-Sleep -Milliseconds 150
        $i++
    }
    Write-Host "`b " -NoNewline
}

# Function to assess ping quality
function Get-PingAssessment {
    param([double]$ping)
    if ($ping -lt 20) { return "Excellent" }
    elseif ($ping -lt 50) { return "Good" }
    elseif ($ping -lt 100) { return "Average" }
    else { return "Poor" }
}

# Function to assess jitter quality
function Get-JitterAssessment {
    param([double]$jitter)
    if ($jitter -lt 5) { return "Excellent" }
    elseif ($jitter -lt 10) { return "Good" }
    elseif ($jitter -lt 20) { return "Average" }
    else { return "Poor" }
}

Write-Host "Starting Speedtest..." -ForegroundColor $colorTitle

# Run the speedtest with JSON output in a background job
$job = Start-Job -ScriptBlock { & $using:SpeedtestPath --format=json }
Show-Spinner -Message "Running Speedtest..." -Condition { (Get-Job -Id $job.Id).State -eq 'Running' }
$resultOutput = Receive-Job -Job $job -Wait
Remove-Job $job

try {
    $result = $resultOutput | ConvertFrom-Json
}
catch {
    Write-Host "Error parsing JSON output: $_" -ForegroundColor $colorError
    exit 1
}

# Extract metrics from the result
$server = $result.server
$ping = $result.ping
$download = $result.download
$upload = $result.upload

# Calculate assessments for ping and jitter
$pingAssessment = Get-PingAssessment -ping $ping.latency
$jitterAssessment = Get-JitterAssessment -jitter $ping.jitter

# Display results with enhanced coloring and announcements
Write-Host "`nConnected to:" -ForegroundColor $colorTitle -NoNewline
Write-Host " $($server.sponsor) ($($server.name), $($server.country)) [ID: $($server.id)]" -ForegroundColor $colorValue
Write-Host "Ping:" -ForegroundColor $colorMetric -NoNewline
Write-Host " $($ping.latency) ms" -ForegroundColor $colorValue -NoNewline
Write-Host " ($pingAssessment)" -ForegroundColor $colorMetric -NoNewline
Write-Host " | Jitter:" -ForegroundColor $colorMetric -NoNewline
Write-Host " $($ping.jitter) ms" -ForegroundColor $colorValue -NoNewline
Write-Host " ($jitterAssessment)" -ForegroundColor $colorMetric

$downloadMbps = [math]::Round($download.bandwidth / 125000, 2)
$uploadMbps = [math]::Round($upload.bandwidth / 125000, 2)
Write-Host "Download Speed:" -ForegroundColor $colorMetric -NoNewline
Write-Host " $downloadMbps Mbps" -ForegroundColor $colorValue
Write-Host "Upload Speed:" -ForegroundColor $colorMetric -NoNewline
Write-Host " $uploadMbps Mbps" -ForegroundColor $colorValue

# Define CSV file path for historical data
$csvPath = "$PSScriptRoot\speedtest_history.csv"

# Log current result to the CSV with assessments
$now = Get-Date
$record = [PSCustomObject]@{
    Date             = $now
    ServerID         = $server.id
    Sponsor          = $server.sponsor
    Location         = "$($server.name), $($server.country)"
    Ping             = $ping.latency
    PingAssessment   = $pingAssessment
    Jitter           = $ping.jitter
    JitterAssessment = $jitterAssessment
    Download         = $downloadMbps
    Upload           = $uploadMbps
}
if (-not (Test-Path $csvPath)) {
    $record | Export-Csv -Path $csvPath -NoTypeInformation
}
else {
    $record | Export-Csv -Path $csvPath -Append -NoTypeInformation
}

Write-Host "Speedtest Complete!" -ForegroundColor $colorTitle

# Function to display ASCII graph with color
function Show-SpeedGraph {
    param(
        [string]$Title,
        [string]$Metric,
        [string]$Color
    )
    Write-Host "`n${Title}:" -ForegroundColor $colorTitle
    Write-Host "====================" -ForegroundColor $Color

    if (-not (Test-Path $csvPath)) {
        Write-Host "No historical data found." -ForegroundColor $colorError
        return
    }

    $data = Import-Csv $csvPath | Sort-Object Date
    if ($data.Count -eq 0) {
        Write-Host "No historical data to display." -ForegroundColor $colorError
        return
    }

    # Get max value for scaling
    $maxValue = ($data | Measure-Object -Property $Metric -Maximum).Maximum
    if ($maxValue -eq 0) { $maxValue = 1 } # Prevent divide by zero

    $maxBarWidth = 50  # Max width for the graph

    foreach ($record in $data) {
        $dateStr = "{0:yyyy-MM-dd HH:mm}" -f ([datetime]$record.Date)
        $value = [double]$record.$Metric

        # Normalize bar length
        $barLength = [math]::Round(($value / $maxValue) * $maxBarWidth)
        if ($barLength -lt 1) { $barLength = 1 } # Ensure at least 1 block is shown

        $bar = "█" * $barLength
        Write-Host "$dateStr :" -ForegroundColor $colorMetric -NoNewline
        Write-Host " $bar" -ForegroundColor $Color -NoNewline
        Write-Host " $value Mbps" -ForegroundColor $colorValue
    }
}

# Display Download & Upload Speed Graphs if historical data exists
if (Test-Path $csvPath) {
    $history = Import-Csv $csvPath
    if ($history.Count -gt 0) {
        Show-SpeedGraph -Title "Historical Download Speeds" -Metric "Download" -Color $colorGraph
        Show-SpeedGraph -Title "Historical Upload Speeds" -Metric "Upload" -Color $colorGraph
    }
}

# Analyze historical data to find the best server if there is more than one record
if (Test-Path $csvPath) {
    $history = Import-Csv $csvPath
    if ($history.Count -gt 1) {
        $bestServer = $history | Sort-Object Ping, Download -Descending | Select-Object -First 1
        Write-Host "`nBest historical server:" -ForegroundColor $colorTitle -NoNewline
        Write-Host " $($bestServer.Sponsor) ($($bestServer.Location)) [ID: $($bestServer.ServerID)]" -ForegroundColor $colorValue
        
        $repeatTest = Read-Host "Do you want to retest using this server? (Y/N)"
        if ($repeatTest -match "^[Yy]") {
            Write-Host "Repeating test with best server ($($bestServer.ServerID))..." -ForegroundColor $colorTitle
            $bestServerID = $bestServer.ServerID
            $job = Start-Job -ScriptBlock { & $using:SpeedtestPath --server-id=$using:bestServerID --format=json }
            Show-Spinner -Message "Running Speedtest with best server..." -Condition { (Get-Job -Id $job.Id).State -eq 'Running' }
            $resultOutput = Receive-Job -Job $job -Wait
            Remove-Job $job
            try {
                $result = $resultOutput | ConvertFrom-Json
                $pingAssessment = Get-PingAssessment -ping $result.ping.latency
                $jitterAssessment = Get-JitterAssessment -jitter $result.ping.jitter
                Write-Host "New test results with best server:" -ForegroundColor $colorTitle
                Write-Host "Ping:" -ForegroundColor $colorMetric -NoNewline
                Write-Host " $($result.ping.latency) ms" -ForegroundColor $colorValue -NoNewline
                Write-Host " ($pingAssessment)" -ForegroundColor $colorMetric -NoNewline
                Write-Host " | Jitter:" -ForegroundColor $colorMetric -NoNewline
                Write-Host " $($result.ping.jitter) ms" -ForegroundColor $colorValue -NoNewline
                Write-Host " ($jitterAssessment)" -ForegroundColor $colorMetric
                Write-Host "Download Speed:" -ForegroundColor $colorMetric -NoNewline
                Write-Host " $([math]::Round($result.download.bandwidth / 125000, 2)) Mbps" -ForegroundColor $colorValue
                Write-Host "Upload Speed:" -ForegroundColor $colorMetric -NoNewline
                Write-Host " $([math]::Round($result.upload.bandwidth / 125000, 2)) Mbps" -ForegroundColor $colorValue
                
                # Log the retest result
                $now = Get-Date
                $record = [PSCustomObject]@{
                    Date             = $now
                    ServerID         = $result.server.id
                    Sponsor          = $result.server.sponsor
                    Location         = "$($result.server.name), $($result.server.country)"
                    Ping             = $result.ping.latency
                    PingAssessment   = $pingAssessment
                    Jitter           = $result.ping.jitter
                    JitterAssessment = $jitterAssessment
                    Download         = [math]::Round($result.download.bandwidth / 125000, 2)
                    Upload           = [math]::Round($result.upload.bandwidth / 125000, 2)
                }
                $record | Export-Csv -Path $csvPath -Append -NoTypeInformation
            }
            catch {
                Write-Host "Error running second test: $_" -ForegroundColor $colorError
            }
        }
    }
}else {
        Write-Host "Not enough historical data to analyze for best server." -ForegroundColor $colorError
    }

# End of script