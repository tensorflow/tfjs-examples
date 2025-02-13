class TextUI {
    constructor(str, x, y, rx, ry, col, normalSize, hoverSize, hoverCol, w, h) {
        if (!hoverCol) {
            this.hoverCol = null;
        } else {
            this.hoverCol = hoverCol;
        }
        this.col = col;
        this.normalSize = normalSize;
        this.hoverSize = hoverSize;
        this.currentSize = this.normalSize;
        this.str = str;
        this.x = x;
        this.y = y;
        this.rx = rx;
        this.w = w;
        this.h = h;
        this.ry = ry - h;
        this.alpha = 255;
        this.startDisappearing = false;
        this.startReappearing = false;
    }
    show() {
        if (this.startReappearing) {
            this.alpha += 2;
            if (this.alpha > 255) {
                this.startReappearing = false;
                this.alpha = 255;
            }
        } else if (this.startDisappearing) {
            this.alpha -= 2;
            if (this.alpha < 0) {
                this.alpha = 0;
                this.startDisappearing = false;
            }
        }
        noStroke();
        fill(this.col, this.alpha);
        textSize(this.currentSize);
        textAlign(CENTER);
        text(this.str, this.x, this.y);
        //stroke(255, 0, 0);
        // noFill();
        // strokeWeight(1);
        // rect(this.rx, this.ry, this.w, this.h);
    }
    hover(mX, mY) {
        if (mX > this.rx && mX < this.rx + this.w && mY > this.ry && mY < this.ry + this.h) {
            this.currentSize = this.hoverSize;
            return true;
        } else {
            this.currentSize = this.normalSize;
            return false;
        }
    }
    disappear() {
        this.startDisappearing = true;
        this.startReappearing = false;
    }
    reappear() {
        this.startReappearing = true;
        this.startDisappearing = false;
    }
    setStr(_str) {
        this.str = _str;
    }
}