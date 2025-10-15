
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scm
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo 'Setting up Python environment...'
                bat 'python --version'
                bat 'pip --version'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing Python packages...'
                bat 'pip install -r requirements.txt'
            }
        }
        
        stage('Verify Installation') {
            steps {
                echo 'Verifying installed packages...'
                bat 'pip list'
            }
        }
        
        stage('Build') {
            steps {
                echo 'Building Mobile Addiction EEG project...'
                bat 'echo All Python packages installed successfully'
                bat 'python -c "import tensorflow as tf; print(tf.__version__)"'
            }
        }
        
        stage('Deploy') {
            steps {
                echo 'Deployment stage...'
                bat 'echo Project ready for deployment'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed'
        }
    }
}
ENDOFFILE
