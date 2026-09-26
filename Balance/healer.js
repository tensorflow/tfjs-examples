class Healer {
    constructor(x, y, w, h, col) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.col = col;
        this.smallerW = this.w * 0.2;
        this.ySpeed = 5;
    }
    show() {
        rectMode(CENTER);
        noStroke();
        fill(this.col);
        rect(this.x, this.y, this.smallerW, this.h);
        rect(this.x, this.y, this.w, this.smallerW);
        for (let i = 0; i < 8; i++) {
            let tempSmallerW = map(i, 0, 7, this.smallerW, this.w * 0.7);
            let tempAlpha = map(i, 0, 7, 100, 40);
            fill(this.col, tempAlpha);
            rect(this.x, this.y, tempSmallerW, this.h);
            rect(this.x, this.y, this.w, tempSmallerW);
        }
    }
    update() {
        this.y += SPEED;
    }
    checkCollision(car) {
        let distFromCenters = dist(car.pos.x, car.pos.y, this.x, this.y);
        if (distFromCenters <= (this.w * 0.5 + car.r)) {
            car.takeDamage(10);
            return true;
        }
        return false;
    }
}