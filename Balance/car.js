class Car {
    constructor(carSide, xPositions, y, col) {
        this.carSide = carSide;
        this.spacing = (width * 0.5) / 2;
        this.xPositions = xPositions;
        this.pos = createVector(this.xPositions[0], y);
        this.r = 10;
        this.col = col;
        this.health = 100;
        this.vel = createVector(0, 0);
        this.acc = createVector(0, 0);
        this.maxSpeed = 20;
        this.startSeeking = false;
        this.seekingPos = null;
        this.currentIndex = 0;
        this.damageAlpha = 0;
        this.health = 100;

    }
    show() {
        fill(this.col);
        ellipse(this.pos.x, this.pos.y, this.r * 2);
        let minR = this.r;
        let maxR = this.r * 2;
        for (let i = 0; i < 10; i++) {
            let _r = map(i, 0, 10, minR, maxR);
            let _alpha = map(i, 0, 10, 100, 0);
            stroke(this.col, _alpha);
            strokeWeight(6);
            noFill();
            ellipse(this.pos.x, this.pos.y, _r * 2);
        }
        //health bar 
        let healthBarLowerY = map(this.health, 0, 100, 0, height * 0.3);
        let healthBarX = (this.carSide == "left") ? width * 0.02 : width * 0.94;
        let healthBarUpperY = 0;
        let healthBarHeight = healthBarLowerY - healthBarUpperY;
        let healthBarC = (this.carSide == "left") ? 0 : 255;
        fill(healthBarC, 200);
        rect(healthBarX, healthBarUpperY, 8, healthBarHeight, 20);
        // //score text 
        // textSize(40);
        // let textCol = (this.carSide == "left") ? 0 : 255;
        // textFont(gameFont);
        // fill(textCol);
        // let textX = (this.carSide == "left") ? width * 0.4 : width * 0.55;
        // if (this.carSide == "left") {
        //     textAlign(LEFT);
        // }

        //damage rect 
        rectMode(CORNER);
        if (this.carSide == "left") {
            fill(0, this.damageAlpha);
            rect(0, 0, width * 0.5, height);
        } else {
            fill(255, this.damageAlpha);
            rect(width * 0.5, 0, width * 0.5, height);
        }
    }
    takeDamage(val) {
        this.damageAlpha = 200;
        this.health += val;
        this.health = constrain(this.health, 0, 100);
    }
    seek(target) {
        //console.log('seeek');
        let desired = p5.Vector.sub(target, this.pos);
        let d = desired.mag();

        desired.normalize();
        if (d < this.spacing) {
            desired.mult(map(d, 0, this.spacing, 0, this.maxSpeed));
        } else {
            desired.mult(this.maxSpeed);
        }
        let steer = p5.Vector.sub(desired, this.vel);
        steer.limit(0.6);
        this.applyForce(steer);
        if (d <= 5) {
            this.startSeeking = false;
            this.seekingPos = null;
            this.vel.mult(0);
            this.acc.mult(0);
            //console.log("Stop seeking!!!!!!!!!");
            //console.log(this.vel);
        }

    }
    update(game) {
        if (this.startSeeking) {
            this.seek(this.seekingPos);
        }
        this.vel.add(this.acc);
        this.pos.add(this.vel);
        this.acc.mult(0);
        this.damageAlpha -= 1;
        if (this.damageAlpha < 0) {
            this.damageAlpha = 0;
        }
    }
    applyForce(f) {
        this.acc.add(f);
    }
    move(moveDir) {
        if (moveDir == "left") {
            this.moveLeft();
        } else if (moveDir == "right") {
            this.moveRight();
        }
    }
    moveRight() {
        if (this.startSeeking) {
            return;
        }
        if (this.currentIndex == 1) {
            return;
        }
        this.currentIndex++;
        this.seekingPos = createVector(this.xPositions[this.currentIndex], this.pos.y);
        this.startSeeking = true;
    }
    moveLeft() {
        if (this.currentIndex == 0) {
            return;
        }
        this.currentIndex--;
        this.seekingPos = createVector(this.xPositions[this.currentIndex], this.pos.y);
        this.startSeeking = true;
    }
}