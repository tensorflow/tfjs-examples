class Block {
    constructor(x, y, w, h, col) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.yvel = 5;
        this.col = col;
    }
    show() {
        rectMode(CENTER);
        noStroke();
        fill(this.col);
        rect(this.x, this.y, this.w, this.h);
        let minW = this.w;
        let minH = this.h;
        let maxW = this.w * 2;
        let maxH = this.h * 2;
        for (let i = 0; i < 5; i++) {
            let _w = map(i, 0, 5, minW, maxW);
            let _h = map(i, 0, 5, minH, maxH);
            let _alpha = map(i, 0, 5, 100, 0);
            noFill();
            stroke(this.col, _alpha);
            strokeWeight(8);
            rect(this.x, this.y, _w, _h);
        }
    }
    update() {
        this.y += SPEED;

    }
    checkCollision(car) {
        let distFromCenters = dist(car.pos.x, car.pos.y, this.x, this.y);
        if (distFromCenters <= (this.w * 0.5 + car.r)) {
            car.takeDamage(-10);
            return true;
        }
        return false;
    }
}