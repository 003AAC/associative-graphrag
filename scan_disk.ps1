$dirs = Get-ChildItem 'C:\' -Directory -Force -ErrorAction SilentlyContinue
foreach ($d in $dirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size/1MB, 0)
    Write-Output "$($d.Name)`t$sizeMB"
}
