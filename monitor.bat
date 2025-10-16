@echo off
echo =========================================
echo Mobile Addiction App - Monitoring
echo =========================================
echo.

echo [1/5] Kubernetes Status:
echo =========================================
kubectl get all
echo.

echo [2/5] Pod Status:
echo =========================================
kubectl get pods -o wide
echo.

echo [3/5] Docker Stats (Snapshot):
echo =========================================
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo.

echo [4/5] Application Logs (Last 10 lines):
echo =========================================
kubectl logs -l app=mobile-addiction --tail=10
echo.

echo [5/5] Service Status:
echo =========================================
kubectl get service mobile-addiction-service
echo.

echo =========================================
echo Monitoring Complete!
echo =========================================
pause
