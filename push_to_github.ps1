Write-Host "Pushing VisionAI to https://github.com/Entangled-mind/VisionAI.git ..." -ForegroundColor Cyan
git push -u origin master
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccessfully pushed to GitHub! View your repository at:" -ForegroundColor Green
    Write-Host "https://github.com/Entangled-mind/VisionAI" -ForegroundColor Yellow
} else {
    Write-Host "`nPush failed. Please ensure you have created the empty repository at:" -ForegroundColor Red
    Write-Host "https://github.com/new?name=VisionAI" -ForegroundColor Yellow
}
