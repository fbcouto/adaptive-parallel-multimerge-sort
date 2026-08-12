# Varre valores de CORANK_SPLIT_FACTOR em src\multimerge.rs, reconstroi em
# release e roda so' Scenario_Random e Scenario_Sawtooth_1000 pra cada
# valor -- sao os dois cenarios que passam pelo merge hibrido co-rank e
# tem mais chance de reagir a esse ajuste. Salva a saida de cada valor num
# arquivo separado em .\sweep_results\ pra comparar depois.
#
# Uso: roda a partir da raiz do projeto (onde fica Cargo.toml):
#   .\sweep_corank_factor.ps1
#
# O arquivo original e' restaurado no final (ou se o script for
# interrompido), entao e' seguro rodar mesmo sem commitar antes -- mas se
# usa git, vale commitar antes de qualquer jeito.

$values = @(2, 4, 8, 16, 32, 64)
$srcFile = "src\multimerge.rs"
$backupFile = "src\multimerge.rs.bak"
$outDir = "sweep_results"

if (-not (Test-Path $srcFile)) {
    Write-Error "Nao achei $srcFile -- rode este script a partir da raiz do projeto."
    exit 1
}

Copy-Item $srcFile $backupFile -Force
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

try {
    foreach ($v in $values) {
        Write-Host ""
        Write-Host "=== CORANK_SPLIT_FACTOR = $v ===" -ForegroundColor Cyan

        (Get-Content $backupFile) -replace 'const CORANK_SPLIT_FACTOR: usize = \d+;', "const CORANK_SPLIT_FACTOR: usize = $v;" |
            Set-Content $srcFile

        $confirmLine = Select-String -Path $srcFile -Pattern "const CORANK_SPLIT_FACTOR"
        Write-Host "  confirmando: $confirmLine"

        $randomLog = Join-Path $outDir "factor_${v}_random.txt"
        Write-Host "  rodando Scenario_Random..."
        cargo bench -- "Scenario_Random" 2>&1 | Tee-Object -FilePath $randomLog | Out-Null

        $sawtoothLog = Join-Path $outDir "factor_${v}_sawtooth.txt"
        Write-Host "  rodando Scenario_Sawtooth_1000..."
        cargo bench -- "Scenario_Sawtooth_1000" 2>&1 | Tee-Object -FilePath $sawtoothLog | Out-Null

        Write-Host "  salvo em $randomLog e $sawtoothLog"
    }
}
finally {
    Copy-Item $backupFile $srcFile -Force
    Remove-Item $backupFile
    Write-Host ""
    Write-Host "Arquivo original restaurado (CORANK_SPLIT_FACTOR de volta ao valor original)." -ForegroundColor Green
}

Write-Host ""
Write-Host "Feito. Pra comparar rapido, roda:" -ForegroundColor Yellow
Write-Host '  Select-String -Path .\sweep_results\*.txt -Pattern "2_MultiMerge_Stable|time:"'
Write-Host "e olha qual valor de fator deu o menor tempo em cada cenario."
