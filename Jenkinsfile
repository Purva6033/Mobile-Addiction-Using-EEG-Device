
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing packages...'
                bat 'pip install -r requirements.txt'
                bat 'pip install flask'
            }
        }
        
        stage('Deploy') {
            steps {
                echo 'Starting Flask app...'
                dir('mobile_Addiction_model/mobile_Addiction_model') {
                    bat 'start /B python app.py'
                }
                echo 'App started on http://localhost:5000'
            }
        }
    }
    
    post {
        success {
            echo 'Deployment successful!'
        }
    }
}
