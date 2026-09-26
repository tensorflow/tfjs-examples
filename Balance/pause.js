class Pause {
    constructor() {
        this.leftX = width * 0.45;
        this.rightX = width * 0.52;
        this.h = 50;
        this.x = width * 0.4;
        this.y = height * 0.9;
        this.w = width * 0.6 - width * 0.4;
        this.ph = this.h - 10;
        this.pw = 10;
    }

    show() {
        fill(0);
        rect(this.leftX, this.y + 10, this.pw, this.ph);
        fill(255);
        rect(this.rightX, this.y + 10, this.pw, this.ph);
    }
}