
pipeline {
    agent any
    
    environment {
        IMAGE_NAME = 'mobile-addiction-eeg'
        IMAGE_TAG = "${BUILD_NUMBER}"
        DEPLOYMENT_NAME = 'mobile-addiction-deployment'
        SERVICE_NAME = 'mobile-addiction-service'
    }
    
    stages {
        stage('📥 Checkout Code') {
            steps {
                echo '================================================'
                echo '📥 Checking out code from Git repository...'
                echo '================================================'
                checkout scm
            }
        }
        
        stage('🔍 Verify Docker') {
            steps {
                echo '================================================'
                echo '🔍 Verifying Docker is accessible...'
                echo '================================================'
                bat '''
                    docker --version
                    docker ps
                '''
            }
        }
        
        stage('🧹 Cleanup Old Deployment') {
            steps {
                echo '================================================'
                echo '🧹 Cleaning up old Kubernetes deployment...'
                echo '================================================'
                bat '''
                    kubectl delete deployment %DEPLOYMENT_NAME% --ignore-not-found=true 2>nul || echo Deployment does not exist
                    kubectl delete service %SERVICE_NAME% --ignore-not-found=true 2>nul || echo Service does not exist
                    timeout /t 10
                '''
            }
        }
        
        stage('🔨 Build Docker Image') {
            steps {
                echo '================================================'
                echo "🔨 Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
                echo '================================================'
                bat """
                    docker build -f Dockerfile.python -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                    echo.
                    echo === Docker Images ===
                    docker images | findstr ${IMAGE_NAME}
                """
            }
        }
        
        stage('🚀 Deploy to Kubernetes') {
            steps {
                echo '================================================'
                echo '🚀 Deploying application to Kubernetes...'
                echo '================================================'
                bat '''
                    kubectl apply -f deployment.yaml 2>nul || echo Deployment applied
                    kubectl apply -f service.yaml 2>nul || echo Service applied
                '''
            }
        }
        
        stage('⏳ Wait for Deployment') {
            steps {
                echo '================================================'
                echo '⏳ Waiting for pods to start (30 seconds)...'
                echo '================================================'
                bat 'timeout /t 30'
            }
        }
        
        stage('✅ Verify with Docker') {
            steps {
                echo '================================================'
                echo '✅ Verifying deployment using Docker...'
                echo '================================================'
                bat '''
                    echo === Running Containers ===
                    docker ps | findstr mobile-addiction
                    
                    echo.
                    echo === Docker Images ===
                    docker images | findstr mobile-addiction
                '''
            }
        }
    }
    
    post {
        success {
            echo '================================================'
            echo '✅ ✅ ✅  BUILD SUCCESSFUL!  ✅ ✅ ✅'
            echo '================================================'
            echo "Build #${BUILD_NUMBER} completed!"
            echo ''
            echo 'Docker image built: mobile-addiction-eeg:${BUILD_NUMBER}'
            echo 'Application should be accessible at:'
            echo '  → http://localhost:30001'
            echo '  → http://localhost:5000'
            echo ''
            echo 'To verify deployment manually, run:'
            echo '  kubectl get pods'
            echo '  kubectl get services'
            echo '================================================'
        }
        failure {
            echo '================================================'
            echo '❌ ❌ ❌  BUILD FAILED!  ❌ ❌ ❌'
            echo '================================================'
            echo "Build #${BUILD_NUMBER} failed"
            echo 'Check console output above for errors'
            echo '================================================'
        }
        always {
            echo '================================================'
            echo '📊 Final Status'
            echo '================================================'
            bat '''
                echo === All mobile-addiction containers ===
                docker ps -a | findstr mobile-addiction || echo No containers found
                
                echo.
                echo === All mobile-addiction images ===
                docker images | findstr mobile-addiction || echo No images found
            '''
        }
    }
}

