import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';

import { Merch } from '../data/merch';
import { MerchService } from '../data/merch.service';

@Component({
  selector: 'app-merch-display',
  templateUrl: './merch-display.component.html',
  styleUrls: ['./merch-display.component.scss'],
})
export class MerchDisplayComponent implements OnInit {
  merch: Merch[] = [];

  constructor(
    private route: ActivatedRoute,
    private merchService: MerchService,
    private location: Location
  ) {}

  ngOnInit(): void {
    this.route.params.subscribe((routeParams) => {
      this.getMerch(routeParams.category);
    });
  }

  getMerch(category: string): void {
    this.merchService
      .getMerchList(category)
      .then((merch) => (this.merch = merch));
  }

  goBack(): void {
    this.location.back();
  }
}
