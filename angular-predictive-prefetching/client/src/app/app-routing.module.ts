import { Routes } from '@angular/router';
import { MerchDisplayComponent } from './merch-display/merch-display.component';

export const routes: Routes = [
  { path: '', component: MerchDisplayComponent },
  { path: ':category', component: MerchDisplayComponent },
];
