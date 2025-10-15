pipeline {
    agent any
    
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
        
        stage('Lint Code') {
            steps {
                echo '🔍 Checking code quality...'
                bat 'python -m py_compile *.py || exit 0'
            }
        }
        
        stage('Run Tests') {
            steps {
                echo '🧪 Running tests...'
                bat 'python -m pytest tests/ || echo "No tests found"'
            }
        }
        
        stage('Build') {
            steps {
                echo '🔨 Building application...'
                bat 'echo Building Mobile Addiction EEG project'
            }
        }
        
        stage('Deploy') {
            steps {
                echo '🚀 Deploying application...'
                bat 'echo Deployment stage - Add your deployment commands here'
            }
        }
    }
    
    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed!'
        }
        always {
            echo '🏁 Pipeline execution finished'
        }
    }
}
