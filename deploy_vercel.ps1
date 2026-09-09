# VisionAI — One-Click Vercel Deployment Script
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  VisionAI — Deploying Live Shareable Website to Vercel   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if npm is installed
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Node.js / npm is not found in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "`nStep 1: Checking Vercel CLI..." -ForegroundColor Yellow
if (-not (Get-Command vercel -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Vercel CLI globally..." -ForegroundColor Green
    npm install -g vercel
}

Write-Host "`nStep 2: Deploying web app to Vercel..." -ForegroundColor Yellow
Write-Host "Follow the interactive login prompt if this is your first time deploying." -ForegroundColor Cyan
vercel --prod

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Deployment Complete! Your site is live and shareable!   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
