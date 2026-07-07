$paths = @('C:\Program Files','C:\Program Files (x86)','C:\Users\华硕\AppData\Local','C:\Users\华硕\AppData\Roaming','D:\','F:\')
foreach ($p in $paths) {
    if (Test-Path $p) {
        Get-ChildItem $p -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'coze|Coze|扣子' } | Select-Object -ExpandProperty FullName
    }
}
