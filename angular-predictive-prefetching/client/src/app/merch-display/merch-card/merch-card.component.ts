import { Component, Input } from '@angular/core';

import { Merch } from 'src/app/data/merch';

@Component({
  selector: 'app-merch-card',
  templateUrl: './merch-card.component.html',
  styleUrls: ['./merch-card.component.scss'],
})
export class MerchCardComponent {
  @Input() merch: Merch;

  get imageUrl() {
    if (!this.merch) return undefined;
    return `https://firebasestorage.googleapis.com/v0/b/merch-store-daa40.appspot.com/o/${this.merch.id}.webp?alt=media`;
  }
}
