import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NavigationComponent } from './navigation/navigation.component';
import { LayoutModule } from '@angular/cdk/layout';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material-experimental/mdc-button';
import { MatListModule } from '@angular/material-experimental/mdc-list';
import { MatCardModule } from '@angular/material-experimental/mdc-card';
import { LogoComponent } from './navigation/logo/logo.component';
import { MatExpansionModule } from '@angular/material/expansion';
import { MerchDisplayComponent } from './merch-display/merch-display.component';
import { MerchCardComponent } from './merch-display/merch-card/merch-card.component';
import { RouterModule } from '@angular/router';
import { routes } from './app-routing.module';

@NgModule({
  declarations: [
    AppComponent,
    NavigationComponent,
    LogoComponent,
    MerchDisplayComponent,
    MerchCardComponent,
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    LayoutModule,
    MatExpansionModule,
    MatToolbarModule,
    MatCardModule,
    MatButtonModule,
    MatSidenavModule,
    MatIconModule,
    MatListModule,
    RouterModule.forRoot(routes),
  ],
  bootstrap: [AppComponent],
})
export class AppModule {}
