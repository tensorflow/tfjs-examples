

class Button {
    constructor(x, y, w, h, img) {
        this.img = img;
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.alpha = 0;
        this.increaseAlpha = false;
    }
    show() {
        image(this.img, this.x, this.y, this.w, this.h);
        //fill(255, this.alpha);
        noFill();
        strokeWeight(8);
        stroke(0);
        rect(this.x, this.y, this.w, this.h);
        if (this.increaseAlpha) {
            this.alpha += 2;
        } else {
            this.alpha -= 2;
        }
        this.alpha = constrain(this.alpha, 0, 255);
    }
    hover(mX, mY) {
        if (mX >= this.x && mX < this.x + this.w && mY >= this.y && mY < this.y + this.h) {
            noFill();
            stroke(0);
            strokeWeight(4);
            rect(this.x, this.y, this.w, this.h, 20);
        }
    }
}