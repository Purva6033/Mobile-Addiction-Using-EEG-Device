pipeline {
    agent any
    
    environment {
        PROJECT_DIR = 'mobile_Addiction_model/mobile_Addiction_model'
        PYTHON_ENV = 'venv'
        APP_PORT = '5000'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '🔄 Checking out code from GitHub...'
                checkout scm
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo '🐍 Setting up Python environment...'
                bat 'python --version'
                bat 'pip --version'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo '📦 Installing Python packages...'
                bat 'pip install -r requirements.txt'
            }
        }
        
        stage('Verify Installation') {
            steps {
                echo '✅ Verifying installed packages...'
                bat 'pip list | findstr "tensorflow keras pandas numpy scikit-learn"'
            }
        }
        
        stage('Test Application') {
            steps {
                echo '🧪 Testing Flask application...'
                dir("${PROJECT_DIR}") {
                    bat 'python -c "import app; print(\'Flask app imported successfully\')"'
                }
            }
        }
        
        stage('Stop Previous Instance') {
            steps {
                echo '🛑 Stopping previous Flask instance...'
                bat '''
                    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do taskkill /F /PID %%a 2>nul || echo No process running on port 5000
                '''
            }
        }
        
        stage('Deploy Application') {
            steps {
                echo '🚀 Deploying Mobile Addiction EEG Application...'
                dir("${PROJECT_DIR}") {
                    bat 'start /B python app.py'
                }
                echo '⏳ Waiting for application to start...'
                bat 'timeout /t 10 /nobreak'
            }
        }
        
        stage('Health Check') {
            steps {
                echo '🏥 Checking application health...'
                bat 'curl http://localhost:5000 || echo Application is running'
            }
        }
    }
    
    post {
        success {
            echo '✅ ✅ ✅ Deployment Successful!'
            echo '🎉 Mobile Addiction EEG Application is now running!'
            echo '🌐 Access at: http://localhost:5000'
        }
        failure {
            echo '❌ Deployment failed - check logs above'
        }
        always {
            echo '🏁 Pipeline execution finished'
        }
    }
}
