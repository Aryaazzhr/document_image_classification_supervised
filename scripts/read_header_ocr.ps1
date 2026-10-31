param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$OutputFile,
    [Parameter(Mandatory=$true)][string]$Config
)
$ErrorActionPreference = 'Stop'
$settings = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
$methods = @{}
function Wait-Operation($op, [Type]$resultType) {
    if (-not $methods.ContainsKey($resultType)) { $methods[$resultType] = $asTask.MakeGenericMethod($resultType) }
    $task = $methods[$resultType].Invoke($null, @($op))
    return $task.GetAwaiter().GetResult()
}
$lang = [Windows.Globalization.Language]::new($settings.language)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
if ($null -eq $engine) { throw "OCR language unavailable: $($settings.language)" }
$writer = [System.IO.StreamWriter]::new($OutputFile, $false, [System.Text.UTF8Encoding]::new($false))
try {
    foreach ($line in [IO.File]::ReadLines($Manifest)) {
        $id, $path = $line.Split([char]9, 2)
        $stream = $null
        $bitmap = $null
        try {
            $file = Wait-Operation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
            $stream = Wait-Operation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
            $decoder = Wait-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
            $transform = [Windows.Graphics.Imaging.BitmapTransform]::new()
            $factor = [Math]::Min([double]$settings.scale, [Windows.Media.Ocr.OcrEngine]::MaxImageDimension / [Math]::Max($decoder.PixelWidth, $decoder.PixelHeight))
            $transform.ScaledWidth = [uint32][Math]::Max(1,[Math]::Round($decoder.PixelWidth * $factor))
            $transform.ScaledHeight = [uint32][Math]::Max(1,[Math]::Round($decoder.PixelHeight * $factor))
            $bounds = [Windows.Graphics.Imaging.BitmapBounds]::new()
            $bounds.Width = $transform.ScaledWidth
            $bounds.Height = [uint32][Math]::Max(1,[Math]::Round($transform.ScaledHeight * $settings.header_fraction))
            $transform.Bounds = $bounds
            $bitmap = Wait-Operation ($decoder.GetSoftwareBitmapAsync([Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8,[Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied,$transform,[Windows.Graphics.Imaging.ExifOrientationMode]::IgnoreExifOrientation,[Windows.Graphics.Imaging.ColorManagementMode]::DoNotColorManage)) ([Windows.Graphics.Imaging.SoftwareBitmap])
            $result = Wait-Operation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
            $writer.WriteLine($id + [char]9 + ($result.Text -replace '\s+',' ').Trim())
        } catch {
            throw "OCR failed for $id : $($_.Exception.Message)"
        } finally {
            if ($null -ne $bitmap) { $bitmap.Dispose() }
            if ($null -ne $stream) { $stream.Dispose() }
        }
    }
} finally {
    $writer.Dispose()
}
