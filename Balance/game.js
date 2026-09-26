class Game {
    constructor() {
        this.startGameText = new TextUI("START GAME", width * 0.5, height * 0.5, 0, height * 0.5, 0, 30, 40, 0, width, 50);
        this.pauseText = new TextUI("Press P for pause", width * 0.5, height * 0.65, 0, 0, 0, 30, 30, 0, 0, 0);
        this.currentHightScore = 0;
        this.backgroundAlpha = 200;
        this.disappearBackground = false;
        this.startGame = false;
        this.playerXPositions = [];
        this.startSpawning = false;
        this.volumePng = loadImage("")
        this.firstSpawnTime = 5;
        let spacing = (width * 0.5) / 2;
        for (let i = 0; i < 4; i++) {
            this.playerXPositions.push(spacing * 0.5 + i * spacing);
        }
        this.player = new Player(this.playerXPositions);
        this.blockManager = new BlockManager(this.playerXPositions);
        this.paused = false;
        this.restartButton = new Button(width * 0.3, width * 0.45, 100, 100, loadImage("imgs/restart_transparent.png"));
        this.gameIsOver = false;
        this.healerManager = new HealerManager();
        this.playerScore = 0;
        this.playerScoreText = new TextUI("--" + this.playerScore + "--", width * 0.5, height * 0.1, 0, 0, 0, 40, 0, 0, 0, 0, 0);
    }
    render() {
        rectMode(CENTER);
        //leftLanes 
        stroke(0);
        strokeWeight(4);
        let spacing = (width * 0.5) / 2;
        for (let i = 0; i < 2; i++) {
            if (i == 0)
                continue;
            let x = i * spacing;
            line(x, 0, x, height);
        }
        //rightLanes
        stroke(255);
        spacing = (width * 0.5) / 2;
        for (let i = 0; i < 2; i++) {
            if (i == 0)
                continue;
            let x = width * 0.5 + i * spacing;
            line(x, 0, x, height);
        }
        fill(255, this.backgroundAlpha);
        noStroke();
        rectMode(CORNER);
        rect(0, 0, width, height);


        this.player.show();
        this.healerManager.show();
        this.blockManager.show();
        textFont(gameFont);
        this.startGameText.show();
        this.pauseText.show();
        textAlign(CENTER);
        this.playerScoreText.show();
        if (this.playerScoreText >= 10000) {
            this.playerScoreText.setStr(this.playerScore);
        } else if (this.playerScore >= 1000) {
            this.playerScoreText.setStr("-" + this.playerScore);
        } else if (this.playerScore >= 100) {
            this.playerScoreText.setStr("--" + this.playerScore);
        } else if (this.playerScore >= 10) {
            this.playerScoreText.setStr("--" + this.playerScore + "-")
        } else {
            this.playerScoreText.setStr("--" + this.playerScore + "--");
        }

    }
    update(mX, mY) {
        if (this.gameIsOver) {
            this.playerScoreText.currentSize *= 1.02;
            this.playerScoreText.y += 2;
            this.playerScoreText.y = constrain(this.playerScoreText.y, height * 0.1, height * 0.2);
            this.playerScoreText.currentSize = constrain(this.playerScoreText.currentSize, this.playerScoreText.normalSize, 80);
        }
        if (!this.paused) {
            if (this.startGame) {
                if (frameCount % 60 == 0) {
                    this.firstSpawnTime -= 1;
                    if (this.firstSpawnTime <= 0) {
                        this.startSpawning = true;
                    }
                }
            }
            this.startGameText.hover(mX, mY);
            if (this.disappearBackground) {
                this.backgroundAlpha -= 1;
                if (this.backgroundAlpha < 0) {
                    this.backgroundAlpha = 0;
                    this.disappearBackground = false;
                    classifyPose();
                }

            }
            if (this.startGame && !this.gameIsOver) {
                this.player.update();
                this.blockManager.update(this);
                this.blockManager.checkCollisions(this.player);
                if (this.player.leftCar.health <= 0 || this.player.rightCar.health <= 0) {
                    this.gameIsOver = true;
                    this.backgroundAlpha = 200;
                    this.reappearText();
                }
                this.healerManager.update(this.blockManager, this);
                this.healerManager.checkCollisions(this.player);
            }
            if (!this.gameIsOver && this.startSpawning && frameCount % 240 == 0) {
                this.playerScore += 1;
                this.playerScore = constrain(this.playerScore, 0, 10000);
                SPEED *= 1.05;
                this.blockManager.spawnRate = floor(this.blockManager.spawnRate * 0.9);
                this.blockManager.spawnRate = constrain(this.blockManager.spawnRate, 1, 3);
                SPEED = constrain(SPEED, 8, 20);
            }
        }
    }
    disappearText() {
        this.startGameText.disappear();
        this.pauseText.disappear();
        this.disappearBackground = true;
    }
    reappearText() {
        this.startGameText.reappear();
        this.pauseText.reappear();
        this.disappearBackground = false;
    }
    checkMouseEvents(mX, mY) {
        if (this.startGameText.hover(mX, mY)) {
            this.startGame = true;
            this.gameIsOver = false;
            this.player = new Player(this.playerXPositions);
            this.playerScoreText = new TextUI("--" + this.playerScore + "--", width * 0.5, height * 0.1, 0, 0, 0, 40, 0, 0, 0, 0, 0);
            this.blockManager = new BlockManager(this.playerXPositions);
            this.healerManager = new HealerManager();
            this.startSpawning = false;
            this.playerScore = 0;
            this.disappearText();
            SPEED = 8;
        }
    }
    checkKeyboardEvents(key, keyCode) {
        if (key == 'p') {
            this.paused = !this.paused;
            if (this.paused) {
                this.backgroundAlpha = 200;
            } else {
                this.disappearBackground = true;
            }
        }
        if (keyCode == 37) {
            this.player.playerMoveRightCar("left");
        } else if (keyCode == 39) {
            this.player.playerMoveRightCar("right");
        }
        if (key == 'a') {
            this.player.playerMoveLeftCar("left");
        } else if (key == 'd') {
            this.player.playerMoveLeftCar("right");
        }
    }

    moveLeftCar(leftCarPose) {
        if (leftCarPose == "LEFTSIDE") {
            this.player.playerMoveLeftCar("left");
        } else {
            this.player.playerMoveLeftCar("right");
        }
    }
    moveRightCar(rightCarPose) {
        if (rightCarPose == "RIGHTSIDE") {
            this.player.playerMoveRightCar("right");
        } else {
            this.player.playerMoveRightCar("left");
        }
    }
}